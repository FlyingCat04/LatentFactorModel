import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.stats import chi2_contingency
import psycopg2
import psycopg2.extras
from scipy.sparse import lil_matrix

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
        domain_id=None,
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
        self.domain_id = domain_id
        
        # --- Handle requirements safely ---
        if self.connection is None and self.train_mode == 'train' and (ratings is None or items is None):
            raise ValueError("❌ DB connection or ratings/items required for training.")
        if self.connection is None and self.train_mode == 'load':
            raise ValueError("❌ DB connection required for loading model.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("❌ train_mode must be 'train' or 'load'")
        
        # --- LOAD MODE ---
        if self.train_mode == 'load':
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: j for j, i in enumerate(self.items_list)}
            self.load_model_from_db(model_id=model_id)
            if self.model is None or not self.users or not self.items_list:
                print("⚠️ Warning: Model load incomplete (maybe cold start). Continuing...")
        
        # --- TRAIN MODE ---
        else:
            if self.k is None:
                raise ValueError("❌ Parameter 'k' (e.g., 90) is required for training.")
            
            # 1. Load User/Item List first
            self.load_user_item_from_db()
            self.user2idx = {u: idx for idx, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            
            # 2. Load Ratings
            if ratings is None:
                self.load_ratings_from_db()
            else:
                self.ratings, self.test_ratings = self.ratings, []
            
            # 3. Handle Empty Data (Cold Start) vs Normal Flow
            self.n_users = len(self.users)
            self.n_items = len(self.items_list)
            
            if not self.ratings:
                print("⚠️ Warning: No training ratings available. Initializing with Random Weights.")
                self.mu = 3.5
                # Initialize random weights since we can't do UCInit logic without data
                P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                b_u_init_arr = np.zeros(self.n_users)
                b_i_init_arr = np.zeros(self.n_items)
            else:
                # Filter ratings
                original_count = len(self.ratings)
                self.ratings = [(u, i, r) for u, i, r in self.ratings if u in self.user2idx and i in self.item2idx]
                filtered = original_count - len(self.ratings)
                if filtered > 0:
                    print(f"⚠️ Filtered out {filtered} ratings with invalid user/item IDs")
                
                self.ratings_dict = {(u, i): r for u, i, r in self.ratings}
                
                # Check again after filtering
                if not self.ratings:
                     print("⚠️ All ratings filtered out. Using Random Init.")
                     self.mu = 3.5
                     P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                     Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                     b_u_init_arr = np.zeros(self.n_users)
                     b_i_init_arr = np.zeros(self.n_items)
                else:
                    # Perform UCInit Logic
                    self.load_item_categories()
                    self.mu = np.mean([r for _, _, r in self.ratings])
                    self.top_p_users = self._get_top_p_users()
                    self.top_v_users = self._get_top_v_users()
                    
                    P_map, Q_map, b_u_map, b_i_map = self._init_latent_model() 
                    
                    P_init_arr = np.array([P_map[u] for u in self.users])
                    Q_init_arr = np.array([Q_map[i] for i in self.items_list])
                    b_u_init_arr = np.array([b_u_map[u] for u in self.users])
                    b_i_init_arr = np.array([b_i_map[i] for i in self.items_list])

            # 4. Create Model
            # Ensure n_users/n_items at least 1 to avoid Embedding crash
            self.model = self.UCInitModel(
                max(self.n_users, 1), max(self.n_items, 1), self.k, self.mu, 
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
                # Safe copy even if shapes slightly mismatch due to fallback
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
    
    def load_user_item_from_db(self):
        cur = self.connection.cursor()
        try:
            cur.execute("""
                SELECT "Id"
                From "User"
                WHERE "DomainId" = %s
                ORDER BY "Id"
            """, (self.domain_id,))
            rows = cur.fetchall()
            self.users = [str(row[0]) for row in rows]
        except Exception as e:
            print(f"⚠️ Error loading users: {e}")
            self.users = []
        
        try:
            cur.execute("""
                SELECT "Id"
                From "Item"
                WHERE "DomainId" = %s
                ORDER BY "Id"
            """, (self.domain_id,))
            rows = cur.fetchall()
            self.items_list = [str(row[0]) for row in rows]
        except Exception as e:
            print(f"⚠️ Error loading items: {e}")
            self.items_list = []
    
    def load_item_categories(self):
        cur = self.connection.cursor()
        
        category_name = {}
        try:
            cur.execute('SELECT * FROM "Category"')
            rows = cur.fetchall()
            for row in rows:
                category_name[row[0]] = row[1]
        except Exception:
            pass
        
        self.item_categories = defaultdict(list)
        if not self.items_list: return

        chunk_size = 1000
        # Safe conversion to int
        items_int = []
        for x in self.items_list:
            try: items_int.append(int(x))
            except: pass
            
        for i in range(0, len(items_int), chunk_size):
            chunk = items_int[i:i+chunk_size]
            if not chunk: continue
            try:
                cur.execute(
                    'SELECT "ItemId", "CategoryId" FROM "ItemCategory" WHERE "ItemId" = ANY(%s)',
                    (chunk,)
                )
                rows = cur.fetchall()
                for iid, cid in rows:
                    if cid in category_name:
                        self.item_categories[str(iid)].append(category_name[cid])
            except Exception:
                continue
        
    def load_model_from_db(self, model_id):
        cur = self.connection.cursor()
        
        item_factors = {}
        item_biases = {}
        temp_k_from_factors = None

        try:
            cur.execute("""
                SELECT "ItemId", "ItemBias", "ItemFactors"
                FROM "ItemFactor"
                WHERE "ModelId" = %s
                ORDER BY "ItemId"
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
                SELECT "UserId", "UserBias", "UserFactors"
                FROM "UserFactor"
                WHERE "ModelId" = %s
                ORDER BY "UserId"
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
            print("⚠️ Warning: Cannot determine K from loaded factors. Using default.")

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
                
        cur.execute('SELECT "AverageRating" FROM "Model" WHERE "Id"=%s', (model_id,))
        mu_row = cur.fetchone()
        self.mu = float(mu_row[0]) if mu_row and mu_row[0] is not None else 3.5
        
        self.model = self.UCInitModel(
            max(n_users_loaded, 1), max(n_items_loaded, 1), self.k, self.mu,
            P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
        ).to(self.device)
        self.model.eval()

        print(f"✅ Loaded Factors/Biases into PyTorch Model for Prediction.")

    def _get_top_p_users(self):
        if not self.ratings: return []
        self.p = min(self.p, self.n_users)
        user_count = Counter([u for u, _, _ in self.ratings])
        top_users = [u for u, _ in user_count.most_common(self.p)]
        return sorted(top_users)

    def _get_top_v_users(self):
        if not self.ratings: return []
        self.v = min(self.v, self.n_users)
        items = self.items_list
        n_users = self.n_users

        try:
            item_to_cat_string = {item: ",".join(self.item_categories[item]) for item in items}
        except TypeError:
            item_to_cat_string = {
                item: ",".join(self.item_categories[item]) if self.item_categories[item] else ""
                for item in items
            }
        
        categories = sorted(list(set(item_to_cat_string.values())))
        n_categories = len(categories)
        if n_categories == 0:
            shuffled_users = list(self.users)
            np.random.shuffle(shuffled_users)
            return shuffled_users[:self.v]
            
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        item_to_cat_idx = {item: cat_to_idx[cat_str] for item, cat_str in item_to_cat_string.items()}

        category_totals = np.zeros(n_categories, dtype=int)
        for item in items:
            if item in item_to_cat_idx:
                category_totals[item_to_cat_idx[item]] += 1

        user_rated_counts = np.zeros((n_users, n_categories), dtype=int)
        for user, item in self.ratings_dict.keys():
            if item in item_to_cat_idx:
                try:
                    u_idx = self.user2idx[user]
                    c_idx = item_to_cat_idx[item]
                    user_rated_counts[u_idx, c_idx] += 1
                except KeyError:
                    pass 

        user_scores = {}
        contingency_table = np.zeros((2, n_categories), dtype=int) 

        for u_idx, user in enumerate(self.users):
            rated_row = user_rated_counts[u_idx, :]
            
            not_rated_row = category_totals - rated_row

            contingency_table[0, :] = rated_row
            contingency_table[1, :] = not_rated_row
            
            if contingency_table.sum() == 0 or \
            np.any(contingency_table.sum(axis=0) == 0) or \
            np.any(contingency_table.sum(axis=1) == 0):
                chi2_val = 0.0
            else:
                try:
                    chi2_val, _, _, _ = chi2_contingency(contingency_table)
                except ValueError:
                    chi2_val = 0.0
            
            user_scores[user] = chi2_val

        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        return [u for u, _ in sorted_users[:self.v]]

    def _UCInit(self):
        items = self.items_list
        top_v_idx = [self.user2idx[u] for u in self.top_v_users]

        R = lil_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u, i, r in self.ratings:
            if u in self.user2idx and i in self.item2idx:
                R[self.user2idx[u], self.item2idx[i]] = r

        R_top = R[top_v_idx, :].toarray()
        Q_init_matrix = R_top.T

        # Nếu số lượng top_v_users < k, cần pad thêm columns
        if Q_init_matrix.shape[1] < self.k:
            padding = np.zeros((Q_init_matrix.shape[0], self.k - Q_init_matrix.shape[1]), dtype=np.float32)
            Q_init_matrix = np.hstack([Q_init_matrix, padding])
        else:
            Q_init_matrix = Q_init_matrix[:, :self.k]

        Q_map = {}
        for idx, item in enumerate(items):
            vec = Q_init_matrix[idx]
            norm = np.linalg.norm(vec)
            Q_map[item] = vec / norm if norm > 0 else vec

        R_norm = normalize(R, norm='l2', axis=1)
        R_v = R_norm[top_v_idx, :self.n_items]

        P_map = {}
        R_v_T = R_v.T
        for u_idx, user in enumerate(self.users):
            row = R_norm[u_idx, :]
            P_init_vector = row.dot(R_v_T).toarray().flatten()
            if len(P_init_vector) < self.k:
                P_map[user] = np.pad(P_init_vector, (0, self.k - len(P_init_vector)))
            else:
                P_map[user] = P_init_vector[:self.k]

        return P_map, Q_map
    
    def _init_latent_model(self):
        P, Q = self._UCInit()
        b_u = {user: 0.0 for user in self.users}
        b_i = {item: 0.0 for item in self.items_list}
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

    def load_ratings_from_db(self, limit_total=1000000, ratio=1):
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
            rows = []

        if not rows:
            print("⚠️ No rating data loaded from DB (Cold start).")
            self.ratings = []
            self.test_ratings = []
            return

        total = len(rows)
        if total == 0:
            self.ratings = []
            self.test_ratings = []
            return

        train_count = int(total * ratio)
        if train_count == 0 or train_count == total:
            if train_count == 0: train_count = total
            if train_count == total: test_ratings = []

        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]
    
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
            
            # Use ratings to determine "seen" vs "unseen" logic (if any)
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
                    # Save initialized factors even if no ratings
                    idx = self.user2idx.get(user_id)
                    if idx is not None:
                        user_factor_data.append((
                            user_id,
                            0.0,
                            all_user_factors[idx].tolist(),
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
                    idx = self.item2idx.get(item_id)
                    if idx is not None:
                        item_factor_data.append((
                            item_id,
                            0.0,
                            all_item_factors[idx].tolist(),
                            self.model_id
                        ))

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
                
            cur = self.connection.cursor()
            cur.execute("""
                UPDATE "Model" SET "AverageRating"=%s WHERE "Id"=%s 
            """, (float(self.mu), self.model_id))
            
            self.connection.commit()
            print(f"✅ Saved model factors (Users: {len(user_factor_data)}, Items: {len(item_factor_data)})")

        except Exception as e:
            print(f"   ❌ Error saving factors/biases (UPSERT stage): {e}")
            self.connection.rollback()
            
    def predict(self, user, item, p):
        if self.model is None:
            return self.mu if self.mu else 3.5

        self.model.eval()
        with torch.no_grad():
            u_idx_val = self.user2idx.get(user)
            i_idx_val = self.item2idx.get(item)
            
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