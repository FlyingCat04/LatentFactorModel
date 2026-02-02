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

db_conn = None
training_lock = asyncio.Lock()
class TrainRequest(BaseModel):
    domain_id: int
    epochs: int = 500
    pla_epochs: int = 500
    batch_size: int = 256
    tolerance: float = 1e-6
    save_after_train: bool = True
    train_submodels: bool = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_conn
    print("🔄 [Startup] Đang khởi tạo hệ thống...")
    try:
        db_conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Database connected successfully.")
    except Exception as e:
        print(f"❌ Lỗi kết nối Database: {e}")
    yield
    print("🛑 [Shutdown] Đóng kết nối...")
    if db_conn:
        db_conn.close()

app = FastAPI(title="Recommender Training API", lifespan=lifespan)

def ensure_models_exist(cursor, domain_id):
    """Tìm hoặc tạo ID cho bộ 5 model của Domain."""
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

def train_sub_model(ModelClass, name, domain_id, model_id, epochs, batch_size, save, interaction_type_id=0):
    print(f"\n⚡ Bắt đầu train sub-model: {name} (ID: {model_id})...")
    # model = ModelClass(
    #     connection=db_conn,
    #     domain_id=domain_id,
    #     model_id=model_id,
    #     k=90,
    #     train_mode='train' 
    # )
    if "iinit" in name.lower():
        model = ModelClass(
            connection=db_conn,
            domain_id=domain_id,
            model_id=model_id,
            k=90,
            train_mode='train',
            interaction_type_id=interaction_type_id
        )
    else:
        model = ModelClass(
            connection=db_conn,
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
    global db_conn
    try:
        cur = db_conn.cursor()
        model_map = ensure_models_exist(cur, domain_id)
        db_conn.commit()
        print(f"ℹ️ Training context: {model_map}")

        if train_submodels:
            train_sub_model(ReviewRatingModel, "ReviewRating", domain_id, model_map["ReviewRating"], epochs, batch_size, save)
            
            train_sub_model(UEIEModel, "UEIE", domain_id, model_map["UEIE"], epochs, batch_size, save)
            
            # Train UCInit
            train_sub_model(UCInitModel, "UCInit", domain_id, model_map["UCInit"], epochs, batch_size, save)
            
            # Train IInit
            # train_sub_model(IInitModel, "IInit", domain_id, model_map["IInit"], epochs, batch_size, save)
            for i in range(1, 9):
                train_sub_model(IInitModel, f"IInit {i}", domain_id, model_map[f"IInit {i}"], epochs, batch_size, save, interaction_type_id=i)

        print(f"\n🚀 Bắt đầu train PLA (ID: {model_map['PLA']})...")
        pla = PLA(
            connection=db_conn,
            domain_id=domain_id,
            model_ids_map=model_map,
            k=90,
            train_mode='train'
        )
        
        pla.fit(n_epochs=pla_epochs, batch_size=batch_size, tol=tol)
        
        if save:
            print("💾 Saving PLA theta...")
            pla.write_model_to_db()
            
        print("\n🎉🎉🎉 TOÀN BỘ QUÁ TRÌNH TRAINING HOÀN TẤT! 🎉🎉🎉")

    except Exception as e:
        print(f"❌ CRITICAL ERROR in training task: {e}")
        import traceback
        traceback.print_exc()
        db_conn.rollback()

@app.post("/api/train", status_code=202)
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    # 2. Kiểm tra nếu đang có task chạy thì báo bận ngay
    if training_lock.locked():
        raise HTTPException(
            status_code=429, 
            detail="Hệ thống đang bận xử lý một request training khác. Vui lòng thử lại sau."
        )
    
    # 3. Hàm wrapper để tự động giải phóng lock khi xong hoặc lỗi
    async def wrapped_task():
        async with training_lock:
            run_training_task(
                req.domain_id, req.epochs, req.pla_epochs, 
                req.batch_size, req.tolerance, req.save_after_train, 
                req.train_submodels
            )

    background_tasks.add_task(wrapped_task)
    return {"message": f"Pipeline Training started for Domain {req.domain_id}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)