import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
import psycopg2
from tqdm import tqdm
from scipy.sparse import lil_matrix
from psycopg2 import extras

class LatentFactorModel(nn.Module):
    def __init__(
        self,
        db_config=None, # THAY ĐỔI: Nhận db_config
        weight=0.3,
        lam=0.01,
        lr=0.001,
        model_id=4,
        domain_id=None,
        train_mode='train',
        device=None,
        k=90,
        interaction_type_id=0
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.db_config = db_config # THAY ĐỔI
        
        if self.db_config is None:
            raise ValueError("❌ DB config is required for all modes.")
        
        if train_mode not in ['train', 'load']:
            raise ValueError("❌ train_mode must be 'train' or 'load'")
        
        self.weight = weight
        self.lam = lam
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.model = None
        self.optimizer = None
        self.k = k
        self.domain_id = domain_id
        self.interaction_type_id = interaction_type_id
        
        # --- LOAD MODE ---
        if self.train_mode == 'load':
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            self.load_model_from_db(model_id)
            if self.model is None or not self.users or not self.items_list:
                print("⚠️ Warning: Model load incomplete (maybe cold start). Continuing...")
        
        # --- TRAIN MODE ---
        else:
            # 1. Load basic info
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            self.n_users = len(self.users)
            self.n_items = len(self.items_list)

            # 2. Load Ratings
            self.load_ratings_from_db()

            # 3. Handle Empty Data vs Normal Flow
            if not self.ratings:
                print(f"⚠️ Warning: No training ratings available for InteractionTypeId={self.interaction_type_id}. Using Random Init.")
                self.mu = 3.5
                # Initialize random weights to avoid NMF crash on empty matrix
                P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                b_u_init_arr = np.zeros(self.n_users)
                b_i_init_arr = np.zeros(self.n_items)
            
            else:
                if self.interaction_type_id == 0:
                    raise ValueError("❌ interaction_type_id must be greater than 0")
                
                # Filter ratings
                original_count = len(self.ratings)
                self.ratings = [(u, i, r) for u, i, r in self.ratings if u in self.user2idx and i in self.item2idx]
                filtered_count = original_count - len(self.ratings)
                if filtered_count > 0:
                    print(f"⚠️ Filtered out {filtered_count} ratings with invalid user/item IDs")
                
                self.ratings, self.test_ratings = self.ratings, []

                if not self.ratings:
                     print("⚠️ All ratings filtered out. Using Random Init.")
                     self.mu = 3.5
                     P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                     Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                     b_u_init_arr = np.zeros(self.n_users)
                     b_i_init_arr = np.zeros(self.n_items)
                else:
                    # Normal Flow: Build Matrix -> NMF
                    self.rating_dict = {(u, i): r for u, i, r in self.ratings}
                    self.mu = np.mean([r for _, _, r in self.ratings])
                    
                    self.build_interaction_matrix()
                    P_map, Q_map, b_u_map, b_i_map = self.init_latent_model()
                    
                    P_init_arr = P_map
                    Q_init_arr = Q_map.T
                    b_u_init_arr = np.array([b_u_map[u] for u in self.users])
                    b_i_init_arr = np.array([b_i_map[i] for i in self.items_list])

            # 4. Initialize Model (Safe for both empty and populated data)
            self.model = self.IInitModel(
                max(self.n_users, 1), max(self.n_items, 1), self.k, self.mu,
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
                # Safe copy handling
                if P_init is not None and P_init.shape == self.P.weight.shape:
                    self.P.weight.copy_(torch.tensor(P_init, dtype=torch.float32))
                else:
                    nn.init.normal_(self.P.weight, 0, 0.01)

                if Q_init is not None and Q_init.shape == self.Q.weight.shape:
                    self.Q.weight.copy_(torch.tensor(Q_init, dtype=torch.float32))
                else:
                    nn.init.normal_(self.Q.weight, 0, 0.01)

                if b_u_init is not None and len(b_u_init) == n_users:
                    self.b_u.weight.copy_(torch.tensor(b_u_init, dtype=torch.float32).unsqueeze(1))
                else:
                    nn.init.zeros_(self.b_u.weight)

                if b_i_init is not None and len(b_i_init) == n_items:
                    self.b_i.weight.copy_(torch.tensor(b_i_init, dtype=torch.float32).unsqueeze(1))
                else:
                    nn.init.zeros_(self.b_i.weight)
                
        def forward(self, user_idx, item_idx):
            p_u = self.P(user_idx)
            q_i = self.Q(item_idx)
            b_u = self.b_u(user_idx).squeeze(dim=-1)
            b_i = self.b_i(item_idx).squeeze(dim=-1)
            dot_product = (p_u * q_i).sum(dim=1)
            return self.mu + b_u + b_i + dot_product
    
    # THAY ĐỔI: Tự mở và đóng kết nối
    def load_user_item_from_db(self):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT "Id"
                        From "User"
                        WHERE "DomainId" = %s
                        ORDER BY "Id"
                    """, (self.domain_id,))
                    self.users = [str(row[0]) for row in cur.fetchall()]
                    
                    cur.execute("""
                        SELECT "Id"
                        From "Item"
                        WHERE "DomainId" = %s
                        ORDER BY "Id"
                    """, (self.domain_id,))
                    self.items_list = [str(row[0]) for row in cur.fetchall()]
        except Exception as e:
            print(f"⚠️ Error loading users/items: {e}")
            self.users = []
            self.items_list = []
    
    # THAY ĐỔI: Gộp lệnh và tự quản lý kết nối
    def load_model_from_db(self, model_id):
        item_factors = {}
        item_biases = {}
        temp_k_from_factors = None
        user_factors = {}
        user_biases = {}
        mu_val = 3.5

        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT "ItemId", "ItemBias", "ItemFactors"
                        FROM "ItemFactor"
                        WHERE "ModelId" = %s
                        ORDER BY "ItemId"
                    """, (model_id,))
                    for item_id, bias, factors in cur.fetchall():
                        item_id = str(item_id)
                        if item_id not in self.item2idx: continue
                        if bias is not None and factors is not None:
                            item_biases[item_id] = float(bias)
                            item_factors[item_id] = np.array(factors, dtype=float)
                            if temp_k_from_factors is None:
                                temp_k_from_factors = len(item_factors[item_id])

                    cur.execute("""
                        SELECT "UserId", "UserBias", "UserFactors"
                        FROM "UserFactor"
                        WHERE "ModelId" = %s
                        ORDER BY "UserId"
                    """, (model_id,))
                    for user_id, bias, factors in cur.fetchall():
                        user_id = str(user_id)
                        if user_id not in self.user2idx: continue
                        if bias is not None and factors is not None:
                            user_biases[user_id] = float(bias)
                            user_factors[user_id] = np.array(factors, dtype=float)
                            if self.k is None and len(user_factors[user_id]) > 0:
                                self.k = len(user_factors[user_id])

                    cur.execute('SELECT "AverageRating" FROM "Model" WHERE "Id"=%s', (model_id,))
                    mu_row = cur.fetchone()
                    if mu_row and mu_row[0] is not None:
                        mu_val = float(mu_row[0])
                        
        except Exception as e:
            print(f"❌ Error loading model factors/biases: {e}")
            raise

        if temp_k_from_factors is not None:
            self.k = temp_k_from_factors
            print(f"   Inferred K={self.k} from loaded item factors.")
        elif self.k is None:
            print("⚠️ Warning: Cannot determine K from loaded factors. Using default.")

        self.mu = mu_val
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
                
        self.model = self.IInitModel(
            max(n_users_loaded, 1), max(n_items_loaded, 1), self.k, self.mu,
            P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
        ).to(self.device)
        self.model.eval()
    
    # THAY ĐỔI: Tự mở và đóng kết nối
    def load_ratings_from_db(self, limit_total=1000000, ratio=1.0):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT "UserId", "ItemId", 1.0
                        FROM "Interaction"
                        WHERE "DomainId" = %s AND "InteractionTypeId" = %s
                        LIMIT %s
                    """, (self.domain_id, self.interaction_type_id, limit_total))
                    rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error querying interactions: {e}")
            rows = []
        
        if not rows:
            print("⚠️ No interaction data loaded from DB (Cold start).")
            self.ratings = []
            self.test_ratings = []
            return

        total = len(rows)
        train_count = int(total * ratio)
        if train_count == 0 and total > 0: train_count = total
        
        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]
    
    def IInit(self):
        # Only run NMF if we have data
        if self.interaction_matrix.sum() == 0:
            return np.random.rand(self.n_users, self.k), np.random.rand(self.k, self.n_items)
            
        model = NMF(n_components=self.k, init="random", random_state=42)
        P = model.fit_transform(self.interaction_matrix)
        Q = model.components_
        return P, Q

    def build_interaction_matrix(self):
        # user x item
        interaction_matrix = lil_matrix((self.n_users, self.n_items), dtype=np.int8)
        
        # We already loaded ratings in self.ratings, so we can use that instead of querying DB again
        # to ensure consistency with the train set
        for u, i, _ in self.ratings:
             if u in self.user2idx and i in self.item2idx:
                user_idx = self.user2idx[u]
                item_idx = self.item2idx[i]
                interaction_matrix[user_idx, item_idx] = 1
                
        self.interaction_matrix = interaction_matrix
    
    def init_latent_model(self):
        P, Q = self.IInit()
        b_u = {u: 0.0 for u in self.users}
        b_i = {i: 0.0 for i in self.items_list}
        return P, Q, b_u, b_i
    
    def train_model(self, epochs=500, batch_size=256):
        if not self.ratings:
            print("⚠️ No training data. Skipping training."); return
            
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
            
            if abs(prev_avg_loss - avg_total_loss_to_track) < tol:
                print(f"✅ Early stopping at epoch {epoch_num+1}")
                break
            prev_avg_loss = avg_total_loss_to_track

        print("🏁 Training finished.")
        
    # THAY ĐỔI: Tự mở connection chỉ ở lúc thực thi INSERT/UPDATE
    def write_model_to_db(self):
        if self.train_mode != 'train': print("Not in train mode. Skipping save."); return
        if self.db_config is None: raise ValueError("❌ DB config required.")
        if self.model is None: print("⚠️ Model not trained. Cannot save."); return

        self.model.eval()

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

        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    if user_factor_data:
                        sql_user = """
                        INSERT INTO "UserFactor" ("UserId", "UserBias", "UserFactors", "ModelId")
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT ("UserId", "ModelId") DO UPDATE SET 
                            "UserBias" = EXCLUDED."UserBias",
                            "UserFactors" = EXCLUDED."UserFactors";
                        """
                        psycopg2.extras.execute_batch(cur, sql_user, user_factor_data)

                    if item_factor_data:
                        sql_item = """
                        INSERT INTO "ItemFactor" ("ItemId", "ItemBias", "ItemFactors", "ModelId")
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT ("ItemId", "ModelId") DO UPDATE SET
                            "ItemBias" = EXCLUDED."ItemBias",
                            "ItemFactors" = EXCLUDED."ItemFactors";
                        """
                        psycopg2.extras.execute_batch(cur, sql_item, item_factor_data)
                        
                    cur.execute("""
                        UPDATE "Model" SET "AverageRating"=%s WHERE "Id"=%s 
                    """, (float(self.mu), self.model_id))
                    
                conn.commit()
                print(f"✅ Saved model factors (Users: {len(user_factor_data)}, Items: {len(item_factor_data)})")

        except Exception as e:
            print(f"   ❌ Error saving factors/biases (UPSERT stage): {e}")
            
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