import psycopg2
import psycopg2.extras
from tqdm import tqdm
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
        db_config=None, # THAY ĐỔI: Nhận db_config
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

        self.db_config = db_config # THAY ĐỔI
        self.domain_id = domain_id
        self.ratings = [] # Initialize as empty list
        self.test_ratings = None
        self.theta = None
        self.k = k
        self.lambda_reg = lambda_reg
        self.lambda_pri = lambda_pri
        self.lr = lr
        self.model_id = model_ids_map.get("PLA")
        self.train_mode = train_mode.lower()
        self.model_ids_map = model_ids_map
        
        if self.db_config is None:
            raise ValueError("❌ DB config required.")
        if self.domain_id is None:
            raise ValueError("❌ 'domain_id' is required.")
        if model_ids_map is None:
            raise ValueError("❌ 'model_ids_map' is required.")

        # --- Load Sub-models ---
        print("--- Loading Sub-models for PLA ---")
        self.review_model = ReviewModel(
            db_config=self.db_config, # THAY ĐỔI
            train_mode='load',
            model_id=model_ids_map["ReviewRating"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ueie_model = UEIEModel(
            db_config=self.db_config, # THAY ĐỔI
            train_mode='load',
            model_id=model_ids_map["UEIE"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ucinit_model = UCInitModel(
            db_config=self.db_config, # THAY ĐỔI
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
                    db_config=self.db_config, # THAY ĐỔI
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
            if not self.ratings:
                print("⚠️ Warning: No training ratings found for PLA. Model will rely on initial Theta/Random weights.")

    # THAY ĐỔI: Tự mở connection
    def load_user_item_from_db(self):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT "Id" FROM "User" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.users = [str(row[0]) for row in cur.fetchall()]
                    
                    cur.execute('SELECT "Id" FROM "Item" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.items = [str(row[0]) for row in cur.fetchall()]
        except Exception as e:
            print(f"❌ Error querying users/items: {e}")
            self.users = []
            self.items = []

    # THAY ĐỔI: Tự mở connection
    def load_theta_from_db(self):
        model_order_names = ["ReviewRating", "UEIE", "UCInit", "IInit 1", "IInit 2", "IInit 3", "IInit 4", "IInit 5", "IInit 6", "IInit 7", "IInit 8"]
        thetas = []
        expected_dim = 2 * self.k

        print("--- Loading Theta from DB ---")
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    for name in model_order_names:
                        mid = self.model_ids_map.get(name)
                        if mid:
                            cursor.execute('SELECT "LearnableParameters" FROM "Model" WHERE "Id" = %s', (mid,))
                            result = cursor.fetchone()
                        else:
                            result = None
                        
                        # Initialize random small noise
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
                        
                        # print(f"   - {name} (ID {mid}): {status}")
                        thetas.append(arr)
        except Exception as e:
            print(f"❌ DB Connection Error loading theta: {e}. Generating random thetas.")
            thetas = [np.random.randn(expected_dim).astype(np.float32) * 0.1 for _ in model_order_names]

        theta_matrix = np.stack(thetas, axis=0)
        self.theta = nn.Parameter(torch.tensor(theta_matrix, dtype=torch.float32, device=self.device))
        print(f"✅ Theta loaded. Shape: {self.theta.shape}")

    # THAY ĐỔI: Tự mở connection
    def load_ratings_from_db(self, limit_total=1000000):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT "UserId", "ItemId", "Value"
                        FROM "Rating"
                        WHERE "DomainId" = %s AND "Value" IS NOT NULL
                        LIMIT %s
                    """, (self.domain_id, limit_total))
                    rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error querying ratings: {e}")
            rows = []

        if not rows:
            print("⚠️ No rating data loaded from DB (Cold start).")
            self.ratings = []
            return

        all_ratings = [(str(u), str(i), float(r)) for u, i, r in rows]
        
        valid_users = set(self.users)
        valid_items = set(self.items)
        
        self.ratings = []
        for u, i, r in all_ratings:
            if u in valid_users and i in valid_items:
                self.ratings.append((u, i, r))
        
        if not self.ratings:
            print("⚠️ Warning: All ratings were filtered out (invalid user/item IDs).")
            return
            
        print(f"✅ Loaded {len(self.ratings)} training ratings.")

    def forward(self, u, i):
        # ---- 1. Get phi(u, i) from Review Model (Base) ----
        u_idx = self.review_model.user2idx.get(u)
        i_idx = self.review_model.item2idx.get(i)
        
        if u_idx is not None:
            if u_idx < self.review_model.model.P.num_embeddings:
                Pu = self.review_model.model.P(torch.tensor([u_idx], device=self.device)).squeeze(0)
            else:
                 Pu = torch.zeros(self.k, device=self.device)
        else:
            Pu = torch.zeros(self.k, device=self.device) 

        if i_idx is not None:
            if i_idx < self.review_model.model.Q.num_embeddings:
                Qi = self.review_model.model.Q(torch.tensor([i_idx], device=self.device)).squeeze(0)
            else:
                Qi = torch.zeros(self.k, device=self.device)
        else:
            Qi = torch.zeros(self.k, device=self.device) 
            
        phi = torch.cat([Pu, Qi], dim=-1)   # shape [2k]

        # ---- 2. Get predictions from ALL submodels ----
        preds = []
        
        # Helper to safely predict
        def safe_predict(model):
            try:
                if model is None: return self.ucinit_model.mu
                val = model.predict(u, i, 0)
                if isinstance(val, (np.ndarray, list, torch.Tensor)):
                    val = float(val)
                return val
            except Exception:
                return self.ucinit_model.mu 

        # 3 Main Models
        preds.append(torch.tensor(safe_predict(self.review_model), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(safe_predict(self.ueie_model), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(safe_predict(self.ucinit_model), dtype=torch.float32, device=self.device))
        
        # 8 IInit Models
        for idx in range(1, 9):
            val = safe_predict(self.iinit_models[idx])
            preds.append(torch.tensor(val, dtype=torch.float32, device=self.device))

        r_s = torch.stack(preds)   # shape [11]

        # ---- 3. Compute weight logits and softmax ----
        logits = torch.mv(self.theta, phi)   # [11]
        alphas = F.softmax(logits, dim=0)    # [11]

        # ---- 4. Weighted combination ----
        r_hat = torch.sum(alphas * r_s) + self.bias.to(self.device)

        return r_hat, alphas, r_s

    def _build_item_index_cache(self, model):
        if model is None or getattr(model, "model", None) is None:
            idx_tensor = torch.full((len(self.items),), -1, device=self.device, dtype=torch.long)
            valid_mask = torch.zeros(len(self.items), device=self.device, dtype=torch.bool)
            return idx_tensor, valid_mask

        max_item_idx = model.model.Q.num_embeddings
        idx_list = []
        valid_list = []
        for item_id in self.items:
            idx = model.item2idx.get(item_id)
            is_valid = idx is not None and idx < max_item_idx
            idx_list.append(idx if is_valid else -1)
            valid_list.append(is_valid)

        idx_tensor = torch.tensor(idx_list, device=self.device, dtype=torch.long)
        valid_mask = torch.tensor(valid_list, device=self.device, dtype=torch.bool)
        return idx_tensor, valid_mask

    def _predict_single_model_for_user(self, model, user_id, item_idx_tensor, item_valid_mask):
        n_items = len(self.items)
        fallback_mu = float(self.ucinit_model.mu)
        if model is None or getattr(model, "model", None) is None:
            return torch.full((n_items,), fallback_mu, device=self.device, dtype=torch.float32)

        m = model.model
        mu = float(getattr(model, "mu", fallback_mu))
        u_idx = model.user2idx.get(user_id)
        user_valid = u_idx is not None and u_idx < m.P.num_embeddings

        if user_valid:
            u_idx_t = torch.tensor([u_idx], device=self.device, dtype=torch.long)
            p_u = m.P(u_idx_t).squeeze(0)
            b_u = m.b_u(u_idx_t).squeeze(0).squeeze(0)

            pred = torch.full((n_items,), mu + b_u.item(), device=self.device, dtype=torch.float32)
            if item_valid_mask.any():
                valid_item_idx = item_idx_tensor[item_valid_mask]
                q_i = m.Q(valid_item_idx)
                b_i = m.b_i(valid_item_idx).squeeze(dim=-1)
                dot = torch.matmul(q_i, p_u)
                pred[item_valid_mask] = mu + b_u + b_i + dot
        else:
            pred = torch.full((n_items,), mu, device=self.device, dtype=torch.float32)
            if item_valid_mask.any():
                valid_item_idx = item_idx_tensor[item_valid_mask]
                b_i = m.b_i(valid_item_idx).squeeze(dim=-1)
                pred[item_valid_mask] = mu + b_i

        return torch.clamp(pred, 1.0, 5.0)

    def _compute_alphas_for_user(self, user_id, review_item_idx_tensor, review_item_valid_mask):
        n_items = len(self.items)
        u_idx = self.review_model.user2idx.get(user_id)

        if u_idx is not None and u_idx < self.review_model.model.P.num_embeddings:
            p_u = self.review_model.model.P(
                torch.tensor([u_idx], device=self.device, dtype=torch.long)
            ).squeeze(0)
        else:
            p_u = torch.zeros(self.k, device=self.device)

        q_i_full = torch.zeros((n_items, self.k), device=self.device, dtype=torch.float32)
        if review_item_valid_mask.any():
            valid_item_idx = review_item_idx_tensor[review_item_valid_mask]
            q_i_full[review_item_valid_mask] = self.review_model.model.Q(valid_item_idx)

        p_u_full = p_u.unsqueeze(0).expand(n_items, -1)
        phi = torch.cat([p_u_full, q_i_full], dim=1)
        logits = torch.matmul(phi, self.theta.t())
        return F.softmax(logits, dim=1)

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
        if n == 0: return {"loss": 0, "main": 0, "priority": 0, "reg": 0}

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
        if not self.ratings:
            print("⚠️ No ratings to train PLA. Skipping training (Using Initialized Weights).")
            self.save_predictions_to_db()
            return

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

            if n_batches > 0:
                epoch_loss /= n_batches
            
            # print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.6f}")

            if prev_loss is not None and abs(prev_loss - epoch_loss) < tol:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            prev_loss = epoch_loss

        print("✅ Training finished!")
        self.save_predictions_to_db() 
        
    # THAY ĐỔI: Chỉ mở kết nối khi thực sự cần INSERT Batch
    def save_predictions_to_db(self):
        """
        Calculates predictions for ALL users and items and saves to DB using Batch Insert.
        """
        start = time.perf_counter()
        if self.db_config is None:
            return
        if not self.users or not self.items:
            print("⚠️ No users/items found. Skip saving predictions.")
            return

        print("💾 Saving all predictions to DB (Batch Processing)...")
        batch_data = []
        batch_size = 10000
        commit_every_n_batches = 5
        pending_batches = 0
        query = """
            INSERT INTO "Predict" ("UserId", "ItemId", "Value")
            VALUES %s
            ON CONFLICT ("UserId", "ItemId") DO UPDATE SET "Value" = EXCLUDED."Value";
        """

        model_sequence = [
            self.review_model,
            self.ueie_model,
            self.ucinit_model,
            self.iinit_models[1],
            self.iinit_models[2],
            self.iinit_models[3],
            self.iinit_models[4],
            self.iinit_models[5],
            self.iinit_models[6],
            self.iinit_models[7],
            self.iinit_models[8],
        ]
        item_index_cache = [self._build_item_index_cache(model) for model in model_sequence]
        review_item_idx_tensor, review_item_valid_mask = item_index_cache[0]
        items = self.items

        self.eval()
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    execute_values = psycopg2.extras.execute_values
                    users = self.users

                    with torch.inference_mode():
                        for u in tqdm(users, desc="Predicting"):
                            pred_cols = []
                            for model, (item_idx_tensor, item_valid_mask) in zip(model_sequence, item_index_cache):
                                pred_cols.append(
                                    self._predict_single_model_for_user(model, u, item_idx_tensor, item_valid_mask)
                                )

                            r_s = torch.stack(pred_cols, dim=1)
                            alphas = self._compute_alphas_for_user(
                                u,
                                review_item_idx_tensor,
                                review_item_valid_mask,
                            )
                            r_hat = torch.sum(alphas * r_s, dim=1) + self.bias
                            r_hat = torch.clamp(r_hat, 1.0, 5.0)
                            values = r_hat.detach().cpu().tolist()

                            batch_data.extend((u, i, float(v)) for i, v in zip(items, values))

                            while len(batch_data) >= batch_size:
                                current_batch = batch_data[:batch_size]
                                execute_values(
                                    cur,
                                    query,
                                    current_batch,
                                    page_size=batch_size,
                                )
                                del batch_data[:batch_size]
                                pending_batches += 1

                                if pending_batches >= commit_every_n_batches:
                                    conn.commit()
                                    pending_batches = 0

                    if batch_data:
                        execute_values(
                            cur,
                            query,
                            batch_data,
                            page_size=len(batch_data),
                        )
                    conn.commit()
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")
            return

        end = time.perf_counter()
        elapsed = end - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"Saving predict time: {elapsed:.2f}s ({minutes}m {seconds:.2f}s)")
        print("✅ Done saving predictions.")
    
    def predict(self, user, item, p):
        # Only used for single prediction inference
        self.eval()
        with torch.no_grad():
            r_hat, _, _ = self.forward(user, item)
            return r_hat.item()

    # THAY ĐỔI: Tự mở connection
    def write_model_to_db(self):
        if self.db_config is None: return
        self.eval()

        theta_values = self.theta.detach().cpu().numpy() 
        model_order_names = ["ReviewRating", "UEIE", "UCInit", "IInit 1", "IInit 2", "IInit 3", "IInit 4", "IInit 5", "IInit 6", "IInit 7", "IInit 8"]
        data_to_insert = []
        
        for idx, name in enumerate(model_order_names):
            mid = self.model_ids_map.get(name)
            if mid:
                theta_list = theta_values[idx].tolist()
                data_to_insert.append((theta_list, mid))

        try:
            if data_to_insert:
                with psycopg2.connect(**self.db_config) as conn:
                    with conn.cursor() as cur:
                        query = 'UPDATE "Model" SET "LearnableParameters" = %s, "ModifiedAt" = NOW() WHERE "Id" = %s;'
                        psycopg2.extras.execute_batch(cur, query, data_to_insert)
                        conn.commit()
                        print(f"💾 Saved {len(data_to_insert)} theta vectors to DB.")
        except Exception as e:
            print(f"❌ Error saving model thetas: {e}")