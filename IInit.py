import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
import psycopg2
from tqdm import tqdm
from scipy.sparse import lil_matrix

class LatentFactorModel(nn.Module):
    def __init__(
        self,
        connection,
        k=90,
        weight=0.3,
        lam=0.01,
        lr=0.001,
        model_id=4,
        train_mode='train',
        device=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if connection is None:
            raise ValueError("❌ DB connection is required for all modes.")
        
        if train_mode not in ['train', 'load']:
            raise ValueError("❌ train_mode must be 'train' or 'load'")
        
        self.connection = connection
        self.k = k
        self.weight = weight
        self.lam = lam
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.model = None
        self.optimizer = None
        
        if self.train_mode == 'load':
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            self.load_model_from_db(model_id)
            if self.model is None or not self.users or not self.items_list:
                raise ValueError("❌ Failed to load model state completely.")
        else:
            self.load_ratings_from_db()
            if not self.ratings:
                raise ValueError("❌ No training ratings available.")
            
            users_in_train_set = set(u for u, _, _ in self.ratings)
            items_in_train_set = set(i for _, i, _ in self.ratings)
            users_in_train = list(users_in_train_set)
            items_in_train = list(items_in_train_set)
            
            if not users_in_train or not items_in_train:
                raise ValueError("❌ No users or items found in the training ratings.")
            
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            self.ratings, self.test_ratings = train_test_split(self.ratings, test_size=0.2, random_state=42)
            self.rating_dict = {(u, i): r for u, i, r in self.ratings}
            self.n_users = len(self.users)
            self.n_items = len(self.items_list)
            self.mu = np.mean([r for _, _, r in self.ratings])
            self.build_interaction_matrix()
            P_map, Q_map, b_u_map, b_i_map = self.init_latent_model()
            
            P_init_arr = P_map
            Q_init_arr = Q_map.T
            b_u_init_arr = np.array([b_u_map[u] for u in self.users])
            b_i_init_arr = np.array([b_i_map[i] for i in self.items_list])

            self.model = self.IInitModel(
                self.n_users, self.n_items, self.k, self.mu,
                P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
            ).to(self.device)
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.lam
            )
    
    class IInitModel(nn.Module):
        def __init__(self, n_users, n_items, k, mu, P_init, Q_init, b_u_init, b_i_init):
            super().__init__()
            
            self.P = nn.Embedding(n_users, k)
            self.Q = nn.Embedding(n_items, k)
            self.b_u = nn.Embedding(n_users, 1)
            self.b_i = nn.Embedding(n_items, 1)
            self.mu = mu
            
            with torch.no_grad():
                self.P.weight.copy_(torch.tensor(P_init, dtype=torch.float32))
                self.Q.weight.copy_(torch.tensor(Q_init, dtype=torch.float32))
                self.b_u.weight.copy_(torch.tensor(b_u_init, dtype=torch.float32).unsqueeze(1))
                self.b_i.weight.copy_(torch.tensor(b_i_init, dtype=torch.float32).unsqueeze(1))
                
        def forward(self, user_idx, item_idx):
            p_u = self.P(user_idx)
            q_i = self.Q(item_idx)
            b_u = self.b_u(user_idx).squeeze(dim=-1)
            b_i = self.b_i(item_idx).squeeze(dim=-1)
            dot_product = (p_u * q_i).sum(dim=1)
            return self.mu + b_u + b_i + dot_product
    
    def load_user_item_from_db(self):
        cur = self.connection.cursor()
        try:
            cur.execute("""
                SELECT "UserID"
                From "User"
                ORDER BY "UserID"
            """, ())
            rows = cur.fetchall()
        except Exception as e:
            raise
            
        self.users = [str(row[0]) for row in rows]
        
        try:
            cur.execute("""
                SELECT "ItemID"
                From "Item"
                ORDER BY "ItemID"
            """, ())
            rows = cur.fetchall()
        except Exception as e:
            raise
        
        self.items_list = [str(row[0]) for row in rows]
    
    def load_model_from_db(self, model_id):
        cur = self.connection.cursor()
        
        item_factors = {}
        item_biases = {}
        temp_k_from_factors = None
        
        try:
            cur.execute("""
                SELECT "ItemID", "ItemBias", "ItemFactors"
                FROM "ItemFactor"
                WHERE "ModelID" = %s
                ORDER BY "ItemID"
            """, (model_id,))
            rows_items = cur.fetchall()
        except Exception as e:
            print(f"❌ Error loading items/factors: {e}")
            raise
        
        for item_id, bias, factors in rows_items:
            item_id = str(item_id)
            if item_id not in self.item2idx:
                continue
            
            if bias is not None and factors is not None:
                item_biases[item_id] = float(bias)
                item_factors[item_id] = np.array(factors, dtype=float)
                if temp_k_from_factors is None:
                    temp_k_from_factors = len(item_factors[item_id])
                    
        if temp_k_from_factors is not None:
            self.k = temp_k_from_factors
            print(f"   Inferred K={self.k} from loaded item factors.")
                
        user_factors = {}
        user_biases = {}
        try:
            cur.execute("""
                SELECT "UserID", "UserBias", "UserFactors"
                FROM "UserFactor"
                WHERE "ModelID" = %s
                ORDER BY "UserID"
            """, (model_id,))
            rows_users = cur.fetchall()
        except Exception as e:
            print(f"❌ Error loading users/factors: {e}")
            raise

        for user_id, bias, factors in rows_users:
            user_id = str(user_id)
            if user_id not in self.user2idx:
                continue
            
            if bias is not None and factors is not None:
                user_biases[user_id] = float(bias)
                user_factors[user_id] = np.array(factors, dtype=float)
                if self.k is None and len(user_factors[user_id]) > 0:
                    self.k = len(user_factors[user_id])
                    print(f" Inferred K={self.k} from loaded user factors.")
        
        if self.k is None:
            raise ValueError("❌ Cannot determine K from loaded factors. Model load failed.")

        n_users_loaded = len(self.users)
        n_items_loaded = len(self.items_list)
        
        P_init_arr = np.zeros((n_users_loaded, self.k), dtype=float)
        Q_init_arr = np.zeros((n_items_loaded, self.k), dtype=float)
        b_u_init_arr = np.zeros(n_users_loaded, dtype=float)
        b_i_init_arr = np.zeros(n_items_loaded, dtype=float)
        
        for user_id, idx in self.user2idx.items():
            if user_id in user_factors:
                P_init_arr[idx] = user_factors[user_id]
            if user_id in user_biases:
                b_u_init_arr[idx] = user_biases[user_id]

        for item_id, idx in self.item2idx.items():
            if item_id in item_factors:
                Q_init_arr[idx] = item_factors[item_id]
            if item_id in item_biases:
                b_i_init_arr[idx] = item_biases[item_id]
                
        cur.execute('SELECT "AverageRating" FROM "Model" WHERE "ModelID"=%s', (model_id,))
        mu_row = cur.fetchone()
        self.mu = float(mu_row[0]) if mu_row and mu_row[0] is not None else 3
        
        self.model = self.IInitModel(
            n_users_loaded, n_items_loaded, self.k, self.mu,
            P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
        ).to(self.device)
        self.model.eval()
    
    def load_ratings_from_db(self, limit_total=1000000, ratio=1.0):
        cur = self.connection.cursor()
        try:
            cur.execute("""
                SELECT "UserID", "ItemID", "RatingValue"
                FROM "Rating"
                ORDER BY RANDOM()
                LIMIT %s
            """, (limit_total,))
            rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error querying ratings: {e}")
            raise
        
        if not rows:
            raise ValueError("❌ No rating data loaded from DB.")

        total = len(rows)
        if total == 0:
            raise ValueError("❌ Zero ratings loaded from DB.")
        
        train_count = int(total * ratio)
        if train_count == 0 or train_count == total:
            if train_count == 0: train_count = total
            if train_count == total: test_ratings = []

        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]
    
    def IInit(self):
        model = NMF(n_components=self.k, init="random", random_state=42)
        P = model.fit_transform(self.interaction_matrix)
        Q = model.components_
        return P, Q

    def build_interaction_matrix(self):
        # user x item
        cur = self.connection.cursor()
        interaction_matrix = lil_matrix((self.n_users, self.n_items), dtype=np.int8)
        cur.execute(
            'SELECT "UserID", "ItemID", "IsRead" FROM "Interaction"'
        )
        rows = cur.fetchall()
        for u, i, read in rows:
            if read == True:
                user_idx = self.user2idx[str(u)]
                item_idx = self.item2idx[str(i)]
                interaction_matrix[user_idx, item_idx] = 1
                
        self.interaction_matrix = interaction_matrix
    
    def init_latent_model(self):
        P, Q = self.IInit()
        b_u = {u: 0.0 for u in self.users}
        b_i = {i: 0.0 for i in self.items_list}
        return P, Q, b_u, b_i
    
    def train_model(self, epochs=500, batch_size=256):
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        self.model.train()
        
        try:
            u_all = torch.tensor([self.user2idx[u] for u, _, _ in self.ratings], device=self.device, dtype=torch.long)
            i_all = torch.tensor([self.item2idx[i] for _, i, _ in self.ratings], device=self.device, dtype=torch.long)
            r_all = torch.tensor([r for _, _, r in self.ratings], device=self.device, dtype=torch.float32)
        except KeyError as e:
            raise KeyError(f"❌ ERROR: User/Item ID missing from index map during tensor creation: {e}")
        
        tol=1e-6
        n = len(r_all)
        prev_avg_loss = float("inf")

        for epoch_num in range(epochs):
            total_mse_sum = 0.0 
            
            perm = torch.randperm(n, device=self.device)
            u_shuf = u_all[perm]
            i_shuf = i_all[perm]
            r_shuf = r_all[perm]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                if start == end: continue
                
                u_batch = u_shuf[start:end] 
                i_batch = i_shuf[start:end]
                r_batch = r_shuf[start:end]
                
                r_hat = self.model(u_batch, i_batch)
                
                err_r = r_batch - r_hat
                main_loss_batch = 0.5 * (err_r**2)
                main_loss_mean = main_loss_batch.mean()
                
                loss = main_loss_mean
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                total_mse_sum += loss.item() * len(r_batch)

            avg_mse_loss = total_mse_sum / n
            
            with torch.no_grad():
                reg_p = torch.sum(self.model.P.weight**2)
                reg_q = torch.sum(self.model.Q.weight**2)
                reg_bu = torch.sum(self.model.b_u.weight**2)
                reg_bi = torch.sum(self.model.b_i.weight**2)
            
            final_reg_loss_total_sum = 0.5 * self.lam * (reg_p + reg_q + reg_bu + reg_bi).item()
            avg_total_loss_to_track = avg_mse_loss + final_reg_loss_total_sum / n
            
            print(f"Epoch {epoch_num+1}/{epochs} | Avg Loss: {avg_total_loss_to_track:.4f}")

            if abs(prev_avg_loss - avg_total_loss_to_track) < tol:
                print(f"✅ Early stopping at epoch {epoch_num+1}")
                break
            prev_avg_loss = avg_total_loss_to_track

        print("🏁 Training finished.")
        
    def write_model_to_db(self):
        if self.train_mode != 'train': print("Not in train mode. Skipping save."); return
        if self.connection is None: raise ValueError("❌ DB connection required.")
        if self.model is None: print("⚠️ Model not trained. Cannot save."); return

        cur = self.connection.cursor()
        self.model.eval()

        try:
            with torch.no_grad():
                all_user_biases = self.model.b_u.weight.squeeze().cpu().numpy()
                all_user_factors = self.model.P.weight.cpu().numpy()
                all_item_biases = self.model.b_i.weight.squeeze().cpu().numpy()
                all_item_factors = self.model.Q.weight.cpu().numpy()

            user_factor_data = []
            item_factor_data = []
            
            users_in_train = set(u for u, _, _ in self.ratings)
            items_in_train = set(i for _, i, _ in self.ratings)

            for user_id in self.users:
                if user_id in users_in_train:
                    idx = self.user2idx[user_id]
                    user_factor_data.append((
                        user_id,
                        float(all_user_biases[idx]),
                        all_user_factors[idx].tolist(),
                        self.model_id
                    ))
                else:
                    user_factor_data.append((
                        user_id,
                        0.0,
                        [0.0]*self.k,
                        self.model_id
                    ))
            
            for item_id in self.items_list:
                if item_id in items_in_train:
                    idx = self.item2idx[item_id]
                    item_factor_data.append((
                        item_id,
                        float(all_item_biases[idx]),
                        all_item_factors[idx].tolist(),
                        self.model_id
                    ))
                else:
                    item_factor_data.append((
                        item_id,
                        0.0,
                        [0.0]*self.k,
                        self.model_id
                    ))

            if user_factor_data:
                sql_user = """
                INSERT INTO "UserFactor" ("UserID", "UserBias", "UserFactors", "ModelID")
                VALUES (%s, %s, %s, %s)
                ON CONFLICT ("UserID", "ModelID") DO UPDATE SET 
                    "UserBias" = EXCLUDED."UserBias",
                    "UserFactors" = EXCLUDED."UserFactors";
                """
                psycopg2.extras.execute_batch(cur, sql_user, user_factor_data)

            if item_factor_data:
                sql_item = """
                INSERT INTO "ItemFactor" ("ItemID", "ItemBias", "ItemFactors", "ModelID")
                VALUES (%s, %s, %s, %s)
                ON CONFLICT ("ItemID", "ModelID") DO UPDATE SET
                    "ItemBias" = EXCLUDED."ItemBias",
                    "ItemFactors" = EXCLUDED."ItemFactors";
                """
                psycopg2.extras.execute_batch(cur, sql_item, item_factor_data)
                
            cur = self.connection.cursor()
            cur.execute("""
                UPDATE "Model" SET "AverageRating"=%s WHERE "ModelID"=%s 
            """, (float(self.mu), self.model_id))
            
            self.connection.commit()

        except Exception as e:
            print(f"   ❌ Error saving factors/biases (UPSERT stage): {e}")
            self.connection.rollback()
            
    def predict(self, user, item, p):
        if self.model is None:
            return self.mu if self.mu else 3.0

        self.model.eval()
        with torch.no_grad():
            u_idx_val = self.user2idx.get(user)
            i_idx_val = self.item2idx.get(item)
            
            # if p == 1:
            #     print(u_idx_val, i_idx_val)
            if u_idx_val is None:
                if i_idx_val is not None:
                    i_idx = torch.tensor([i_idx_val], device=self.device, dtype=torch.long)
                    b_i = self.model.b_i(i_idx).squeeze().item()
                    prediction = self.mu + b_i
                else:
                    prediction = self.mu
                return np.clip(prediction, 1.0, 5.0)

            if i_idx_val is None:
                u_idx = torch.tensor([u_idx_val], device=self.device, dtype=torch.long)
                b_u = self.model.b_u(u_idx).squeeze().item()
                prediction = self.mu + b_u
                return np.clip(prediction, 1.0, 5.0)

            u_idx = torch.tensor([u_idx_val], device=self.device, dtype=torch.long)
            i_idx = torch.tensor([i_idx_val], device=self.device, dtype=torch.long)
            prediction = self.model(u_idx, i_idx).item()
            
            return np.clip(prediction, 1.0, 5.0)
    
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    DB_NAME = "rsystem"
    USER = "flyingcat2003"
    HOST = "localhost"
    PASSWORD = "Hanly1912a"

    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, 
            user=USER, 
            host=HOST, 
            password=PASSWORD,
            options="-c search_path=dbo,goodreads"
            )
        import psycopg2.extras
        print("✅ Database connected successfully.")
    except Exception as e:
        print(f"❌ DB Connection Error: {e}")
        exit()
        
    try:
        model = LatentFactorModel(
            connection=conn,
            train_mode='train',
            model_id=4,
            k=90,
            lr=0.001,
            lam=0.01
        )
        
        # print(model.predict("b85cd4976dbbb72055cefbb984ebe941", "25074277", 1))
        
        test_ratings_to_use = model.test_ratings if model.test_ratings is not None else []
        if not test_ratings_to_use:
             print("⚠️ No test ratings loaded. Evaluation will be skipped.")

        if model.model is not None:
            print("\n--- Starting Model Training ---")
            model.train_model(epochs=500, batch_size=256)
        else:
            print("❌ Model initialization failed. Skipping training.")

        if model.model is not None:
            print("\n--- Saving Model State ---")
            model.write_model_to_db()
        else:
            print("Skipping save.")
        
        if test_ratings_to_use:
            print("\n--- Evaluating Model ---")

            def compute_rmse(model_instance, ratings_set):
                if not ratings_set: return float('nan')
                squared_error = 0.0
                count = 0
                for user, item, r_ui in tqdm(ratings_set, desc="RMSE Eval"):
                    pred = 0.0
                    if count < 50:
                        pred = model_instance.predict(user, item, 1)
                    else:
                        pred = model_instance.predict(user, item, 0)
                    squared_error += (r_ui - pred) ** 2
                    if count < 50:
                        print(f"   {user}-{item}: true={r_ui}, pred={pred:.2f}")
                        count += 1
                if not ratings_set: return 0.0
                mse = squared_error / len(ratings_set)
                rmse = np.sqrt(mse)
                return rmse

            def compute_mae(model_instance, ratings_set):
                if not ratings_set: return float('nan')
                absolute_error = 0.0
                for user, item, r_ui in tqdm(ratings_set, desc="MAE Eval"):
                    pred = model_instance.predict(user, item)
                    absolute_error += abs(r_ui - pred)
                if not ratings_set: return 0.0
                mae = absolute_error / len(ratings_set)
                return mae

            test_rmse = compute_rmse(model, test_ratings_to_use)
            print(f"RMSE on test set: {test_rmse:.4f}")

            test_mae = compute_mae(model, test_ratings_to_use)
            print(f"MAE on test set: {test_mae:.4f}")

    except ValueError as ve: 
        print(f"\n❌ Initialization/Data Error: {ve}")
    except Exception as e: 
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if conn is not None:
            conn.close()
            print("\nDatabase connection closed.")