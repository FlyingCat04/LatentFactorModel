import psycopg2
import psycopg2.extras
from tqdm import tqdm
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ReviewRating import LatentFactorModel as ReviewModel
    from .UCInit import LatentFactorModel as UCInitModel
    from .UEIE import LatentFactorModel as UEIEModel
    from .IInit import LatentFactorModel as IInitModel
except ImportError:
    from ReviewRating import LatentFactorModel as ReviewModel
    from UCInit import LatentFactorModel as UCInitModel
    from UEIE import LatentFactorModel as UEIEModel
    from IInit import LatentFactorModel as IInitModel

class PLA(nn.Module):
    def __init__(
        self,
        connection=None, 
        domain_id=None,       
        model_ids_map=None,
        k=90,
        lambda_reg=0.01,
        lambda_pri=0.01,
        lr=0.002,
        model_id=-1,
        train_mode='train',
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.connection = connection
        self.domain_id = domain_id
        self.ratings = None
        self.test_ratings = None
        self.theta = None
        self.k = k
        self.lambda_reg = lambda_reg
        self.lambda_pri = lambda_pri
        self.lr = lr
        self.model_id = model_ids_map.get("PLA")
        self.train_mode = train_mode.lower()
        self.model_ids_map = model_ids_map
        
        if self.connection is None:
            raise ValueError("❌ DB connection required.")
        if self.domain_id is None:
            raise ValueError("❌ 'domain_id' is required.")
        if model_ids_map is None:
            raise ValueError("❌ 'model_ids_map' is required.")

        # --- Load Sub-models ---
        print("--- Loading Sub-models for PLA ---")
        self.review_model = ReviewModel(
            connection=self.connection,
            train_mode='load',
            model_id=model_ids_map["ReviewRating"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ueie_model = UEIEModel(
            connection=self.connection,
            train_mode='load',
            model_id=model_ids_map["UEIE"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ucinit_model = UCInitModel(
            connection=self.connection,
            train_mode='load',
            model_id=model_ids_map["UCInit"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01
        )

        # Init 8 IInit models (Index 1->8)
        self.iinit_models = [None] * 9
        for i in range(1, 9):
            try:
                self.iinit_models[i] = IInitModel(
                    connection=self.connection,
                    train_mode='load',
                    model_id=model_ids_map[f"IInit {i}"],
                    domain_id=self.domain_id,
                    k=90,
                    lr=0.001,
                    lam=0.01,
                    interaction_type_id=i
                )
            except Exception as e:
                print(f"⚠️ Warning: Failed to load IInit {i}: {e}. Will fallback to mean.")
                self.iinit_models[i] = None

        self.load_user_item_from_db()

        # Freeze submodels
        for param in self.review_model.parameters(): param.requires_grad = False
        for param in self.ueie_model.parameters(): param.requires_grad = False
        for param in self.ucinit_model.parameters(): param.requires_grad = False
        for i in range(1, 9):
            if self.iinit_models[i]:
                for param in self.iinit_models[i].parameters():
                    param.requires_grad = False

        # --- Learnable Parameters ---
        self.num_models = 11  # Review(1) + UEIE(1) + UCInit(1) + IInit(8)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Load Theta
        self.load_theta_from_db()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lambda_reg)
        self.loss_fn = nn.MSELoss()

        if self.train_mode == 'train':
            self.load_ratings_from_db()

    def load_user_item_from_db(self):
        cur = self.connection.cursor()
        try: 
            cur.execute('SELECT "Id" FROM "User" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
            self.users = [str(row[0]) for row in cur.fetchall()]
            
            cur.execute('SELECT "Id" FROM "Item" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
            self.items = [str(row[0]) for row in cur.fetchall()]
        except Exception as e:
            print(f"❌ Error querying users/items: {e}")
            raise

    def load_theta_from_db(self):
        cursor = self.connection.cursor()
        model_order_names = ["ReviewRating", "UEIE", "UCInit", "IInit 1", "IInit 2", "IInit 3", "IInit 4", "IInit 5", "IInit 6", "IInit 7", "IInit 8"]
        thetas = []
        expected_dim = 2 * self.k

        print("--- Loading Theta from DB ---")
        for name in model_order_names:
            mid = self.model_ids_map[name]
            cursor.execute('SELECT "LearnableParameters" FROM "Model" WHERE "Id" = %s', (mid,))
            result = cursor.fetchone()
            
            arr = np.random.randn(expected_dim).astype(np.float32) * 0.1
            status = "🆕 NULL/Empty -> Random Init"

            if result and result[0] is not None:
                try:
                    db_arr = np.array(result[0], dtype=np.float32)
                    if db_arr.size == expected_dim:
                        arr = db_arr
                        status = "✅ Loaded"
                    else:
                        status = f"⚠️ Size Mismatch ({db_arr.size} vs {expected_dim}) -> Re-init"
                except Exception:
                    status = "❌ Parse Error -> Re-init"
            print(f"   - {name} (ID {mid}): {status}")
            thetas.append(arr)

        theta_matrix = np.stack(thetas, axis=0)
        self.theta = nn.Parameter(torch.tensor(theta_matrix, dtype=torch.float32, device=self.device))
        print(f"✅ Theta loaded. Shape: {self.theta.shape}")

    def load_ratings_from_db(self, limit_total=1000000):
        cur = self.connection.cursor()
        try:
            cur.execute("""
                SELECT "UserId", "ItemId", "Value"
                FROM "Rating"
                WHERE "DomainId" = %s AND "Value" IS NOT NULL
                LIMIT %s
            """, (self.domain_id, limit_total))
            rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error querying ratings: {e}")
            raise

        if not rows:
            raise ValueError("❌ No rating data loaded from DB.")

        all_ratings = [(str(u), str(i), float(r)) for u, i, r in rows]
        
        # --- FIXED FILTERING LOGIC ---
        # Don't require intersection of submodels. Just require valid System User/Item
        valid_users = set(self.users)
        valid_items = set(self.items)
        
        self.ratings = []
        for u, i, r in all_ratings:
            if u in valid_users and i in valid_items:
                self.ratings.append((u, i, r))
        
        if not self.ratings:
            raise ValueError("❌ No valid ratings found after filtering.")
            
        print(f"✅ Loaded {len(self.ratings)} training ratings.")

    def forward(self, u, i):
        # ---- 1. Get phi(u, i) from Review Model (Base) ----
        # Nếu user/item không có trong Review Model, dùng vector 0 hoặc random
        u_idx = self.review_model.user2idx.get(u)
        i_idx = self.review_model.item2idx.get(i)
        
        if u_idx is not None:
            Pu = self.review_model.model.P(torch.tensor([u_idx], device=self.device)).squeeze(0)
        else:
            Pu = torch.zeros(self.k, device=self.device) # Fallback

        if i_idx is not None:
            Qi = self.review_model.model.Q(torch.tensor([i_idx], device=self.device)).squeeze(0)
        else:
            Qi = torch.zeros(self.k, device=self.device) # Fallback
            
        phi = torch.cat([Pu, Qi], dim=-1)   # shape [2k]

        # ---- 2. Get predictions from ALL submodels (FIXED) ----
        preds = []
        
        # 3 model chính
        preds.append(torch.tensor(self.review_model.predict(u, i, 0), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(self.ueie_model.predict(u, i, 0), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(self.ucinit_model.predict(u, i, 0), dtype=torch.float32, device=self.device))
        
        # 8 IInit Models (Loop 1 -> 8)
        for idx in range(1, 9):
            model = self.iinit_models[idx]
            if model is not None:
                val = model.predict(u, i, 0)
                preds.append(torch.tensor(val, dtype=torch.float32, device=self.device))
            else:
                # Fallback nếu model bị lỗi load hoặc không tồn tại
                preds.append(torch.tensor(self.ucinit_model.mu, dtype=torch.float32, device=self.device))

        r_s = torch.stack(preds)   # shape [11]

        # ---- 3. Compute weight logits and softmax ----
        logits = torch.mv(self.theta, phi)   # [11]
        alphas = F.softmax(logits, dim=0)    # [11]

        # ---- 4. Weighted combination ----
        r_hat = torch.sum(alphas * r_s) + self.bias.to(self.device)

        return r_hat, alphas, r_s

    def compute_loss(self, r_true, r_hat, alphas):
        main_loss = 0.5 * (r_true - r_hat) ** 2
        
        # Regularization on alpha differences (smoothness)
        diff = alphas[1:] - alphas[:-1] # Compare adjacent alphas
        pri_loss = 0.5 * torch.sum(diff ** 2)
        
        reg_loss = 0.5 * torch.sum(self.theta ** 2)
        
        total = main_loss + self.lambda_pri * pri_loss + self.lambda_reg * reg_loss
        return total, main_loss, pri_loss, reg_loss

    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        total_loss, total_main, total_pri, total_reg = 0.0, 0.0, 0.0, 0.0

        for (u, i, r_true) in batch:
            r_true_tensor = torch.tensor(r_true, dtype=torch.float32, device=self.device)
            r_hat, alphas, r_s = self.forward(u, i)
            loss, main_loss, pri_loss, reg_loss = self.compute_loss(r_true_tensor, r_hat, alphas)

            total_loss += loss
            total_main += main_loss
            total_pri += pri_loss
            total_reg += reg_loss

        n = len(batch)
        avg_loss = total_loss / n
        avg_loss.backward()
        self.optimizer.step()

        return {
            "loss": avg_loss.item(),
            "main": (total_main / n).item(),
            "priority": (total_pri / n).item(),
            "reg": (total_reg / n).item()
        }
    
    def fit(self, n_epochs=500, batch_size=256, tol=1e-6):
        self.train()
        n_samples = len(self.ratings)
        n_batches = (n_samples + batch_size - 1) // batch_size
        prev_loss = None

        print(f"🚀 Start training PLA model ({n_epochs} epochs)...")
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            
            # Shuffle ratings
            random.shuffle(self.ratings)

            for b in range(n_batches):
                batch = self.ratings[b * batch_size:(b + 1) * batch_size]
                if not batch: continue
                losses = self.train_step(batch)
                epoch_loss += losses["loss"]

            epoch_loss /= n_batches
            print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.6f}")

            if prev_loss is not None and abs(prev_loss - epoch_loss) < tol:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            prev_loss = epoch_loss

        print("✅ Training finished!")
        # IMPORTANT: Do not run save_predictions loop here automatically.
        # Call save_predictions_to_db() explicitly if needed.
        self.save_predictions_to_db() 
        
    def save_predictions_to_db(self):
        """
        Calculates predictions for ALL users and items and saves to DB using Batch Insert.
        """
        if self.connection is None: return
        print("💾 Saving all predictions to DB (Batch Processing)...")
        cur = self.connection.cursor()
        
        batch_data = []
        batch_size = 2000 # Larger batch size for speed
        
        self.eval()
        with torch.no_grad():
            for u in tqdm(self.users, desc="Predicting"):
                for i in self.items:
                    try:
                        r_hat, _, _ = self.forward(u, i)
                        val = r_hat.item()
                        val = max(1.0, min(5.0, val)) # Clip 1-5
                        batch_data.append((u, i, val))
                        
                        if len(batch_data) >= batch_size:
                            query = """
                                INSERT INTO "Predict" ("UserId", "ItemId", "Value")
                                VALUES %s
                                ON CONFLICT ("UserId", "ItemId") DO UPDATE SET "Value" = EXCLUDED."Value";
                            """
                            psycopg2.extras.execute_values(cur, query, batch_data)
                            self.connection.commit()
                            batch_data = []
                    except Exception:
                        continue
            
            # Insert remaining
            if batch_data:
                query = """
                    INSERT INTO "Predict" ("UserId", "ItemId", "Value")
                    VALUES %s
                    ON CONFLICT ("UserId", "ItemId") DO UPDATE SET "Value" = EXCLUDED."Value";
                """
                psycopg2.extras.execute_values(cur, query, batch_data)
                self.connection.commit()
        print("✅ Done saving predictions.")
    
    def predict(self, user, item, p):
        # Only used for single prediction inference
        self.eval()
        with torch.no_grad():
            r_hat, _, _ = self.forward(user, item)
            return r_hat.item()

    def write_model_to_db(self):
        if self.connection is None: return
        cur = self.connection.cursor()
        self.eval()

        theta_values = self.theta.detach().cpu().numpy() 
        model_order_names = ["ReviewRating", "UEIE", "UCInit", "IInit 1", "IInit 2", "IInit 3", "IInit 4", "IInit 5", "IInit 6", "IInit 7", "IInit 8"]
        data_to_insert = []
        
        for idx, name in enumerate(model_order_names):
            mid = self.model_ids_map[name]
            theta_list = theta_values[idx].tolist()
            data_to_insert.append((theta_list, mid))

        try:
            query = 'UPDATE "Model" SET "LearnableParameters" = %s, "ModifiedAt" = NOW() WHERE "Id" = %s;'
            psycopg2.extras.execute_batch(cur, query, data_to_insert)
            self.connection.commit()
            print(f"💾 Saved {len(data_to_insert)} theta vectors to DB.")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            self.connection.rollback()
        finally:
            cur.close()