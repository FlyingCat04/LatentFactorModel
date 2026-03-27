import asyncio
import traceback
import psycopg2
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import settings

# --- MODEL IMPORTS ---
try:
    from models.PLA import PLA
    from models.ReviewRating import LatentFactorModel as ReviewRatingModel
    from models.UEIE import LatentFactorModel as UEIEModel
    from models.UCInit import LatentFactorModel as UCInitModel
    from models.IInit import LatentFactorModel as IInitModel
except ImportError as e:
    print(f"Import Error: {e}")
    raise e

# --- SYSTEM CONFIGURATION ---
DB_CONFIG = settings.DB_CONFIG
MODEL_TYPES = [
    "PLA", "ReviewRating", "UEIE", "UCInit", 
    "IInit 1", "IInit 2", "IInit 3", "IInit 4", 
    "IInit 5", "IInit 6", "IInit 7", "IInit 8"
]

# Asynchronous queue to handle concurrent training requests safely
training_queue = asyncio.Queue()

class TrainRequest(BaseModel):
    """Data model representing the payload for incoming training requests."""
    domain_id: int
    epochs: int = 500
    pla_epochs: int = 500
    batch_size: int = 256
    tolerance: float = 1e-6
    save_after_train: bool = True
    train_submodels: bool = True

async def worker_process_queue():
    """
    Background worker task that continually listens to the queue and processes 
    training requests sequentially to prevent resource exhaustion and DB connection timeouts.
    """
    while True:
        task_args = await training_queue.get()
        domain_id = task_args['domain_id']
        
        try:
            # Offload the CPU-bound training task to a separate thread
            # to avoid blocking the main asynchronous event loop of FastAPI
            await asyncio.to_thread(
                run_training_task,
                task_args['domain_id'],
                task_args['epochs'],
                task_args['pla_epochs'],
                task_args['batch_size'],
                task_args['tolerance'],
                task_args['save_after_train'],
                task_args['train_submodels']
            )
        except Exception as e:
            print(f"Worker Error - Failed processing domain {domain_id}: {e}")
        finally:
            training_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifecycle, including the background worker."""
    worker_task = asyncio.create_task(worker_process_queue())
    yield
    worker_task.cancel()

# Initialize FastAPI application
app = FastAPI(title="Recommender Training API", lifespan=lifespan)

def ensure_models_exist(cursor, domain_id):
    """
    Queries the database to fetch IDs for all required models within a domain.
    If a model does not exist, it creates a new record and returns the generated ID.
    """
    model_ids = {}
    for name in MODEL_TYPES:
        cursor.execute('SELECT "Id" FROM "Model" WHERE "DomainId" = %s AND "Name" = %s LIMIT 1', (domain_id, name))
        row = cursor.fetchone()
        if row:
            model_ids[name] = row[0]
        else:
            cursor.execute("""
                INSERT INTO "Model" ("Name", "DomainId", "AverageRating", "ModifiedAt")
                VALUES (%s, %s, 0, NOW()) RETURNING "Id"
            """, (name, domain_id))
            model_ids[name] = cursor.fetchone()[0]
    return model_ids

def train_sub_model(ModelClass, name, domain_id, model_id, epochs, batch_size, save, db_config, interaction_type_id=0):
    """Instantiates, trains, and saves a specific sub-model."""
    if "iinit" in name.lower():
        model = ModelClass(
            db_config=db_config,
            domain_id=domain_id,
            model_id=model_id,
            k=90,
            train_mode='train',
            interaction_type_id=interaction_type_id
        )
    else:
        model = ModelClass(
            db_config=db_config,
            domain_id=domain_id,
            model_id=model_id,
            k=90,
            train_mode='train' 
        )

    model.train_model(epochs=epochs, batch_size=batch_size)
    
    if save:
        model.write_model_to_db()

def run_training_task(domain_id, epochs, pla_epochs, batch_size, tol, save, train_submodels):
    """
    Main orchestration logic for the training pipeline.
    Manages a localized database connection to map model IDs, delegates training to 
    sub-models, and ultimately trains the PLA ensemble model.
    """
    conn = None 
    try:
        # 1. Open a temporary database connection exclusively for metadata operations
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            cur = conn.cursor()
            model_map = ensure_models_exist(cur, domain_id)
            conn.commit() 

            # 2. Train individual feature-extraction sub-models
            if train_submodels:
                train_sub_model(ReviewRatingModel, "ReviewRating", domain_id, model_map["ReviewRating"], epochs, batch_size, save, DB_CONFIG)
                train_sub_model(UEIEModel, "UEIE", domain_id, model_map["UEIE"], epochs, batch_size, save, DB_CONFIG)
                train_sub_model(UCInitModel, "UCInit", domain_id, model_map["UCInit"], epochs, batch_size, save, DB_CONFIG)
                
                for i in range(1, 9):
                    train_sub_model(IInitModel, f"IInit {i}", domain_id, model_map[f"IInit {i}"], epochs, batch_size, save, DB_CONFIG, interaction_type_id=i)

            # 3. Train the Priority-based Learning Algorithm (PLA) ensemble
            pla = PLA(
                db_config=DB_CONFIG,
                domain_id=domain_id,
                model_ids_map=model_map,
                k=90,
                train_mode='train'
            )
            
            pla.fit(n_epochs=pla_epochs, batch_size=batch_size, tol=tol)
            
            if save:
                pla.write_model_to_db()
            
            # Commit final changes if all steps succeeded
            conn.commit()

            print(f"[Domain {domain_id}] training done.")

        except Exception as inner_e:
            # Rollback active transaction if an internal logical error occurs
            if conn:
                conn.rollback()
            raise inner_e

    except Exception as e:
        print(f"CRITICAL ERROR in training pipeline for Domain {domain_id}: {e}")
        traceback.print_exc()
        
    finally:
        # Ensure database connection is safely closed regardless of success or failure
        if conn is not None:
            conn.close()

@app.post("/api/train", status_code=202)
async def trigger_training(req: TrainRequest):
    """
    API endpoint to receive and enqueue training requests.
    Returns HTTP 202 Accepted immediately without waiting for the training to finish.
    """
    await training_queue.put({
        'domain_id': req.domain_id,
        'epochs': req.epochs,
        'pla_epochs': req.pla_epochs,
        'batch_size': req.batch_size,
        'tolerance': req.tolerance,
        'save_after_train': req.save_after_train,
        'train_submodels': req.train_submodels
    })
    
    return {
        "message": "Training request queued successfully.", 
        "position_in_queue": training_queue.qsize(),
        "domain_id": req.domain_id
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)