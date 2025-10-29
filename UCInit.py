import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import psycopg2
import psycopg2.extras

class LatentFactorModel(nn.Module):
    def __init__(
        self,
        ratings=None, 
        items=None,
        connection=None, 
        k=90,
        lam=0.01,
        lr=0.001,
        p=1000,
        model_id=2,
        train_mode='train',
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.connection = connection
        self.lam = lam
        self.lr = lr
        self.p = p
        self.v = k
        self.k = k
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.model = None
        self.optimizer = None
        
        if self.connection is None and self.train_mode == 'train' and (ratings is None or items is None):
            raise ValueError("❌ DB connection or ratings/items required for training.")
        if self.connection is None and self.train_mode == 'load':
            raise ValueError("❌ DB connection required for loading model.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("❌ train_mode must be 'train' or 'load'")
        
        if self.train_mode == 'load':
            self.load_full_model_from_db(model_id=model_id)
            if self.model is None or not self.users or not self.items_list:
                raise ValueError("❌ Failed to load model state completely.")
        
        else:
            if self.k is None:
                raise ValueError("❌ Parameter 'k' (e.g., 90) is required for training.")
            
            if ratings is None:
                self.load_ratings_from_db()
            else:
                self.ratings, self.test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
                
            if not self.ratings:
                raise ValueError("❌ No training ratings available.")
            
            users_in_train_set = set(u for u, _, _ in self.ratings)
            items_in_train_set = set(i for _, i, _ in self.ratings)
            users_in_train = list(users_in_train_set)
            items_in_train = list(items_in_train_set)
            
            if not users_in_train or not items_in_train:
                raise ValueError("❌ No users or items found in the training ratings.")
            
            self.users = sorted(list({u for u, _, _ in self.ratings}))
            self.items_list = sorted(list({i for _, i, _ in self.ratings}))
            self.user2idx = {u: idx for idx, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            self.ratings, self.test_ratings = train_test_split(self.ratings, test_size=0.2, random_state=42)
            self.ratings_dict = {(u, i): r for u, i, r in self.ratings}
            self.load_item_categories()
            self.n_users = len(self.users)
            self.n_items = len(self.items_list)
            self.mu = np.mean([r for _, _, r in self.ratings])

        
            self.top_p_users = self._get_top_p_users()
            self.top_v_users = self._get_top_v_users()
        
            P_map, Q_map, b_u_map, b_i_map = self._init_latent_model() 
            
            P_init_arr = np.array([P_map[u] for u in self.users])
            Q_init_arr = np.array([Q_map[i] for i in self.items_list])
            b_u_init_arr = np.array([b_u_map[u] for u in self.users])
            b_i_init_arr = np.array([b_i_map[i] for i in self.items_list])

            self.model = self.UCInitModel(
                self.n_users, self.n_items, self.k, self.mu, 
                P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
            ).to(self.device)
            
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.lam
            )
        
    class UCInitModel(nn.Module):
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
        
    def load_item_categories(self):
        cur = self.connection.cursor()
        
        category_name = {}
        # id, name
        cur.execute('SELECT * FROM "Category"')
        rows = cur.fetchall()
        for row in rows:
            category_name[row[0]] = row[1]
        
        self.item_categories = defaultdict(list)
        # item id, category id
        cur.execute('SELECT * FROM "ItemCategory"')
        rows = cur.fetchall()
        for item in self.items_list:
            for iid, cid in rows:
                if item == iid:
                    self.item_categories[item].append(category_name[cid])
        
    def load_full_model_from_db(self, model_id):
        cur = self.connection.cursor()
        
        items_list_temp = []
        item_factors = {}
        item_biases = {}
        temp_k_from_factors = None

        try:
            cur.execute("""
                SELECT i."ItemID", f."ItemBias", f."ItemFactors"
                FROM "Item" i
                LEFT JOIN "ItemFactor" f ON i."ItemID" = f."ItemID" AND f."Model" = %s
            """, (model_id,))
            rows_items = cur.fetchall()
        except Exception as e:
            print(f"❌ Error loading items/factors: {e}")
            raise
        print(f"   Found {len(rows_items)} items in 'Item' table.")

        for item_id, bias, factors in rows_items:
            items_list_temp.append(item_id)
            
            if bias is not None and factors is not None:
                item_biases[item_id] = float(bias)
                item_factors[item_id] = np.array(factors, dtype=float)
                if temp_k_from_factors is None:
                    temp_k_from_factors = len(item_factors[item_id])

        if temp_k_from_factors is not None:
            self.k = temp_k_from_factors
            print(f"   Inferred K={self.k} from loaded item factors.")

        self.items_list = items_list_temp
        
        users_list_temp = []
        user_factors = {}
        user_biases = {}
        try:
            cur.execute("""
                SELECT u."UserID", f."UserBias", f."UserFactors"
                FROM "User" u
                LEFT JOIN "UserFactor" f ON u."UserID" = f."UserID" AND f."Model" = %s
            """, (model_id,))
            rows_users = cur.fetchall()
        except Exception as e:
            print(f"❌ Error loading users/factors: {e}")
            raise

        for user_id, bias, factors in rows_users:
            users_list_temp.append(user_id)
            
            if bias is not None and factors is not None:
                user_biases[user_id] = float(bias)
                user_factors[user_id] = np.array(factors, dtype=float)
                if self.k is None and len(user_factors[user_id]) > 0:
                    self.k = len(user_factors[user_id])
                    print(f"   Inferred K={self.k} from loaded user factors.")

        if self.k is None:
             raise ValueError("❌ Cannot determine K from loaded factors. Model load failed.")

        self.users = users_list_temp

        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.item2idx = {i: j for j, i in enumerate(self.items_list)}

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
                
        cur.execute('SELECT AVG("RatingValue") FROM "Rating"')
        mu_row = cur.fetchone()
        self.mu = float(mu_row[0]) if mu_row and mu_row[0] is not None else 3
        
        self.model = self.UCInitModel(
            n_users_loaded, n_items_loaded, self.k, self.mu,
            P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
        ).to(self.device)
        self.model.eval()

        with torch.no_grad():
            loaded_params = 0
            for user_id, bias in user_biases.items():
                if user_id in self.user2idx:
                    self.model.b_u.weight[self.user2idx[user_id]] = bias
                    loaded_params +=1
            for item_id, bias in item_biases.items():
                if item_id in self.item2idx:
                    self.model.b_i.weight[self.item2idx[item_id]] = bias
                    loaded_params +=1
            for user_id, factors in user_factors.items():
                 if user_id in self.user2idx:
                    if len(factors) == self.k:
                        self.model.P.weight[self.user2idx[user_id]] = torch.tensor(factors, dtype=torch.float32, device=self.device)
                        loaded_params += self.k
            for item_id, factors in item_factors.items():
                if item_id in self.item2idx:
                    if len(factors) == self.k:
                        self.model.Q.weight[self.item2idx[item_id]] = torch.tensor(factors, dtype=torch.float32, device=self.device)
                        loaded_params += self.k
        print(f"✅ Loaded Factors/Biases into PyTorch Model for Prediction.")

    def _get_top_p_users(self):
        self.p = min(self.p, self.n_users)
        user_count = Counter([u for u, _, _ in self.ratings])
        top_users = [u for u, _ in user_count.most_common(self.p)]
        return sorted(top_users)

    def _get_top_v_users(self):
        self.v = min(self.v, self.n_users)
        items = self.items_list
        categories = sorted(set([",".join(self.item_categories[item]) for item in items]))
        user_scores = {}
        for user in self.users:
            table = np.zeros((2, len(categories)), dtype=int)
            for idx, cat in enumerate(categories):
                for item in items:
                    if cat == ",".join(self.item_categories[item]):
                        if (user, item) in self.ratings_dict:
                            table[0, idx] += 1
                        else:
                            table[1, idx] += 1
            if table.sum() == 0 or np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
                chi2_val = 0
            else:
                chi2_val, _, _, _ = chi2_contingency(table)
            user_scores[user] = chi2_val
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        return [u for u, _ in sorted_users[:self.v]]

    def _UCInit(self):
        items = self.items_list
        top_v_idx = [self.user2idx[u] for u in self.top_v_users]

        R = np.zeros((self.n_users, self.n_items), dtype=float)
        for u, i, r in self.ratings:
            R[self.user2idx[u], self.item2idx[i]] = r

        Q_map = {}
        Q_init_matrix = R[top_v_idx, :].T 
        for idx, item in enumerate(items):
            Q_map[item] = Q_init_matrix[idx]
            
        for item in Q_map:
            norm = np.linalg.norm(Q_map[item])
            if norm > 0:
                Q_map[item] = Q_map[item] / norm
            else:
                 Q_map[item] = Q_map[item] 

        P_map = {}
        R_norm = np.linalg.norm(R, axis=1, keepdims=True)
        R_norm[R_norm == 0] = 1
        R_normalized = R / R_norm 
        R_v = R_normalized[top_v_idx, :]
        P_init_matrix = np.dot(R_normalized, R_v.T) 

        for u_idx, user in enumerate(self.users):
            P_map[user] = P_init_matrix[u_idx]

        return P_map, Q_map
    
    def _init_latent_model(self):
        P, Q = self._UCInit()
        b_u = {user: 0.0 for user in self.users}
        b_i = {item: 0.0 for item in self.items_list}
        
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
            
            print(f"Epoch {epoch_num+1}/{epochs} | Avg Loss: {avg_total_loss_to_track:.4f} (MSE: {avg_mse_loss:.4f}, Reg: {final_reg_loss_total_sum:.2f})")

            if abs(prev_avg_loss - avg_total_loss_to_track) < tol:
                print(f"✅ Early stopping at epoch {epoch_num+1}")
                break
            prev_avg_loss = avg_total_loss_to_track

        print("🏁 Training finished.")


    def predict(self, user, item):
        if self.model is None:
            return self.mu if self.mu else 3.0
             
        self.model.eval()
        with torch.no_grad():
            u_idx = self.user2idx.get(user)
            i_idx = self.item2idx.get(item)
            
            if u_idx is None or i_idx is None:
                return np.clip(self.mu, 1.0, 5.0) 
            
            u_idx_tensor = torch.tensor([u_idx], device=self.device, dtype=torch.long)
            i_idx_tensor = torch.tensor([i_idx], device=self.device, dtype=torch.long)
            
            prediction = self.model(u_idx_tensor, i_idx_tensor).item()
            return np.clip(prediction, 1.0, 5.0)

    def load_ratings_from_db(self, limit_total=1000000, ratio=0.8):
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

        self.ratings = [(u, i, float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(u, i, float(r)) for u, i, r in rows[train_count:]]
    
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

            for user_id in self.users:
                if user_id in self.user2idx:
                    idx = self.user2idx[user_id]
                    if 0 <= idx < len(all_user_biases) and 0 <= idx < len(all_user_factors):
                        user_factor_data.append((
                            user_id,
                            float(all_user_biases[idx]),
                            all_user_factors[idx].tolist(),
                            self.model_id
                        ))
            
            for item_id in self.items_list:
                if item_id in self.item2idx:
                    idx = self.item2idx[item_id]
                    if 0 <= idx < len(all_item_biases) and 0 <= idx < len(all_item_factors):
                        item_factor_data.append((
                            item_id,
                            float(all_item_biases[idx]),
                            all_item_factors[idx].tolist(),
                            self.model_id
                        ))

            if user_factor_data:
                sql_user = """
                INSERT INTO "UserFactor" ("UserID", "UserBias", "UserFactors", "Model")
                VALUES (%s, %s, %s, %s)
                ON CONFLICT ("UserID", "Model") DO UPDATE SET 
                    "UserBias" = EXCLUDED."UserBias",
                    "UserFactors" = EXCLUDED."UserFactors";
                """
                psycopg2.extras.execute_batch(cur, sql_user, user_factor_data)

            if item_factor_data:
                sql_item = """
                INSERT INTO "ItemFactor" ("ItemID", "ItemBias", "ItemFactors", "Model")
                VALUES (%s, %s, %s, %s)
                ON CONFLICT ("ItemID", "Model") DO UPDATE SET
                    "ItemBias" = EXCLUDED."ItemBias",
                    "ItemFactors" = EXCLUDED."ItemFactors";
                """
                psycopg2.extras.execute_batch(cur, sql_item, item_factor_data)
            self.connection.commit()

        except Exception as e:
            print(f"   ❌ Error saving factors/biases (UPSERT stage): {e}")
            self.connection.rollback()
            
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
        conn = psycopg2.connect(dbname=DB_NAME, user=USER, host=HOST, password=PASSWORD)
        import psycopg2.extras
        print("✅ Database connected successfully.")
    except Exception as e:
        print(f"❌ DB Connection Error: {e}")
        exit()
        
    try:
        model = LatentFactorModel(
            connection=conn,
            train_mode='load',
            model_id=2,
            k=90,
            lr=0.001,
            lam=0.01
        )
        
        print(model.predict(1, 720))
        
        # test_ratings_to_use = model.test_ratings if model.test_ratings is not None else []
        # if not test_ratings_to_use:
        #      print("⚠️ No test ratings loaded. Evaluation will be skipped.")

        # if model.model is not None:
        #     print("\n--- Starting Model Training ---")
        #     model.train_model(epochs=500, batch_size=512)
        # else:
        #     print("❌ Model initialization failed. Skipping training.")

        # if model.model is not None:
        #     print("\n--- Saving Model State ---")
        #     model.write_model_to_db()
        # else:
        #     print("Skipping save.")
        
        # if test_ratings_to_use:
        #     print("\n--- Evaluating Model ---")

        #     def compute_rmse(model_instance, ratings_set):
        #         if not ratings_set: return float('nan')
        #         squared_error = 0.0
        #         count = 0
        #         for user, item, r_ui in tqdm(ratings_set, desc="RMSE Eval"):
        #             pred = model_instance.predict(user, item)
        #             squared_error += (r_ui - pred) ** 2
        #             if count < 20:
        #                 print(f"   {user}-{item}: true={r_ui}, pred={pred:.2f}")
        #                 count += 1
        #         if not ratings_set: return 0.0
        #         mse = squared_error / len(ratings_set)
        #         rmse = np.sqrt(mse)
        #         return rmse

        #     def compute_mae(model_instance, ratings_set):
        #         if not ratings_set: return float('nan')
        #         absolute_error = 0.0
        #         for user, item, r_ui in tqdm(ratings_set, desc="MAE Eval"):
        #             pred = model_instance.predict(user, item)
        #             absolute_error += abs(r_ui - pred)
        #         if not ratings_set: return 0.0
        #         mae = absolute_error / len(ratings_set)
        #         return mae

        #     test_rmse = compute_rmse(model, test_ratings_to_use)
        #     print(f"RMSE on test set: {test_rmse:.4f}")

        #     test_mae = compute_mae(model, test_ratings_to_use)
        #     print(f"MAE on test set: {test_mae:.4f}")

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