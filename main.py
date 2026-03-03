import uvicorn
import psycopg2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
from config import settings
import sys
import asyncio

# --- IMPORT ĐẦY ĐỦ CÁC MODEL ---
try:
    from models.PLA import PLA
    from models.ReviewRating import LatentFactorModel as ReviewRatingModel
    from models.UEIE import LatentFactorModel as UEIEModel
    from models.UCInit import LatentFactorModel as UCInitModel
    from models.IInit import LatentFactorModel as IInitModel
except ImportError as e:
    print(f"❌ Import Error: {e}")
    raise e

DB_CONFIG = settings.DB_CONFIG
MODEL_TYPES = ["PLA", "ReviewRating", "UEIE", "UCInit", "IInit 1", "IInit 2", "IInit 3", "IInit 4", "IInit 5", "IInit 6", "IInit 7", "IInit 8"]

training_queue = asyncio.Queue()

class TrainRequest(BaseModel):
    domain_id: int
    epochs: int = 500
    pla_epochs: int = 500
    batch_size: int = 256
    tolerance: float = 1e-6
    save_after_train: bool = True
    train_submodels: bool = True

async def worker_process_queue():
    print("👷 Queue Worker started waiting for tasks...")
    while True:
        task_args = await training_queue.get()
        domain_id = task_args['domain_id']
        
        print(f"🔄 [Queue] Start processing Domain {domain_id}. Pending: {training_queue.qsize()}")
        
        try:
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
            print(f"❌ [Queue] Error processing domain {domain_id}: {e}")
        finally:
            training_queue.task_done()
            print(f"✅ [Queue] Finished Domain {domain_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(worker_process_queue())
    print("🚀 System started.")
    yield
    print("🛑 System shutting down.")
    worker_task.cancel()

app = FastAPI(title="Recommender Training API", lifespan=lifespan)

def ensure_models_exist(cursor, domain_id):
    model_ids = {}
    for name in MODEL_TYPES:
        cursor.execute('SELECT "Id" FROM "Model" WHERE "DomainId" = %s AND "Name" = %s LIMIT 1', (domain_id, name))
        row = cursor.fetchone()
        if row:
            model_ids[name] = row[0]
        else:
            print(f"🆕 Creating new model '{name}' for Domain {domain_id}...")
            cursor.execute("""
                INSERT INTO "Model" ("Name", "DomainId", "AverageRating", "ModifiedAt")
                VALUES (%s, %s, 0, NOW()) RETURNING "Id"
            """, (name, domain_id))
            model_ids[name] = cursor.fetchone()[0]
    return model_ids

# THAY ĐỔI: Truyền db_config thay vì conn
def train_sub_model(ModelClass, name, domain_id, model_id, epochs, batch_size, save, db_config, interaction_type_id=0):
    print(f"\n⚡ Bắt đầu train sub-model: {name} (ID: {model_id})...")
    
    if "iinit" in name.lower():
        model = ModelClass(
            db_config=db_config, # ĐỔI Ở ĐÂY
            domain_id=domain_id,
            model_id=model_id,
            k=90,
            train_mode='train',
            interaction_type_id=interaction_type_id
        )
    else:
        model = ModelClass(
            db_config=db_config, # ĐỔI Ở ĐÂY
            domain_id=domain_id,
            model_id=model_id,
            k=90,
            train_mode='train' 
        )

    model.train_model(epochs=epochs, batch_size=batch_size)
    
    if save:
        print(f"💾 Saving {name}...")
        model.write_model_to_db()
    print(f"✅ Đã xong {name}.")

def run_training_task(domain_id, epochs, pla_epochs, batch_size, tol, save, train_submodels):
    print(f"🔌 [Domain {domain_id}] Initializing models...")
    
    # 1. Chỉ mở kết nối NHANH để lấy Model IDs
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                model_map = ensure_models_exist(cur, domain_id)
            conn.commit()
    except Exception as e:
        print(f"❌ [Domain {domain_id}] Error ensuring models: {e}")
        return

    print(f"ℹ️ [Domain {domain_id}] Context: {model_map}")

    try:
        if train_submodels:
            # Truyền DB_CONFIG vào thay vì conn
            train_sub_model(ReviewRatingModel, "ReviewRating", domain_id, model_map["ReviewRating"], epochs, batch_size, save, DB_CONFIG)
            train_sub_model(UEIEModel, "UEIE", domain_id, model_map["UEIE"], epochs, batch_size, save, DB_CONFIG)
            train_sub_model(UCInitModel, "UCInit", domain_id, model_map["UCInit"], epochs, batch_size, save, DB_CONFIG)
            for i in range(1, 9):
                train_sub_model(IInitModel, f"IInit {i}", domain_id, model_map[f"IInit {i}"], epochs, batch_size, save, DB_CONFIG, interaction_type_id=i)

        print(f"\n🚀 [Domain {domain_id}] Training PLA...")
        pla = PLA(
            db_config=DB_CONFIG, # ĐỔI Ở ĐÂY
            domain_id=domain_id,
            model_ids_map=model_map,
            k=90,
            train_mode='train'
        )
        
        pla.fit(n_epochs=pla_epochs, batch_size=batch_size, tol=tol)
        
        if save:
            print(f"💾 [Domain {domain_id}] Saving PLA theta...")
            pla.write_model_to_db()
        
        print(f"\n🎉 [Domain {domain_id}] TRAINING SUCCESSFUL! 🎉")

    except Exception as e:
        print(f"❌ [Domain {domain_id}] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

@app.post("/api/train", status_code=202)
async def trigger_training(req: TrainRequest):
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