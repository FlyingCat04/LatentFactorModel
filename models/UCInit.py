import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
from scipy.stats import chi2_contingency
import psycopg2
import psycopg2.extras
from scipy.sparse import lil_matrix

class LatentFactorModel(nn.Module):
    """
    Latent Factor Model utilizing User-Category Initialization (UCInit).
    
    This model initializes user and item latent factors by analyzing the correlation 
    between users' historical interactions and item categories using the Chi-Square test,
    subsequently fine-tuning the matrices via a PyTorch embedding layer.
    """
    def __init__(
        self,
        ratings=None, 
        items=None,
        db_config=None, 
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
        
        # Configuration parameters
        self.db_config = db_config 
        self.lam = lam
        self.lr = lr
        self.p = p
        self.v = k
        self.k = k
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.domain_id = domain_id
        
        # Explicit declaration of all instance variables to prevent AttributeError
        self.model = None
        self.optimizer = None
        self.users = []
        self.items_list = []
        self.user2idx = {}
        self.item2idx = {}
        self.ratings = []
        self.test_ratings = []
        self.ratings_dict = {}
        self.item_categories = defaultdict(list)
        self.top_p_users = []
        self.top_v_users = []
        self.mu = 3.5
        self.n_users = 0
        self.n_items = 0
        
        if self.db_config is None:
            raise ValueError("Database configuration is required.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("train_mode must be strictly 'train' or 'load'.")

        self._initialize_system(ratings)

    def _initialize_system(self, input_ratings):
        """Internal method to orchestrate data loading and model initialization."""
        if self.train_mode == 'load':
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: j for j, i in enumerate(self.items_list)}
            self.load_model_from_db(self.model_id)
            
        else: # train_mode == 'train'
            if self.k is None:
                raise ValueError("Latent dimension 'k' is required for training.")
            
            self.load_user_item_from_db()
            self.user2idx = {u: idx for idx, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            
            self.n_users = len(self.users)
            self.n_items = len(self.items_list)
            
            if input_ratings is None:
                self.load_ratings_from_db()
            else:
                self.ratings = input_ratings
                self.test_ratings = []
            
            # Handle cold-start scenario with random initialization if no data is present
            if not self.ratings:
                self.mu = 3.5
                P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                b_u_init_arr = np.zeros(self.n_users)
                b_i_init_arr = np.zeros(self.n_items)
            else:
                # Filter ratings to ensure valid indices
                self.ratings = [(u, i, r) for u, i, r in self.ratings if u in self.user2idx and i in self.item2idx]
                self.ratings_dict = {(u, i): r for u, i, r in self.ratings}
                
                if not self.ratings:
                    self.mu = 3.5
                    P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                    Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                    b_u_init_arr = np.zeros(self.n_users)
                    b_i_init_arr = np.zeros(self.n_items)
                else:
                    self.load_item_categories()
                    self.mu = np.mean([r for _, _, r in self.ratings])
                    
                    # Extract active users and structurally significant users
                    self.top_p_users = self._get_top_p_users()
                    self.top_v_users = self._get_top_v_users()
                    
                    P_map, Q_map, b_u_map, b_i_map = self._init_latent_model() 
                    
                    P_init_arr = np.array([P_map.get(u, np.zeros(self.k)) for u in self.users])
                    Q_init_arr = np.array([Q_map.get(i, np.zeros(self.k)) for i in self.items_list])
                    b_u_init_arr = np.array([b_u_map.get(u, 0.0) for u in self.users])
                    b_i_init_arr = np.array([b_i_map.get(i, 0.0) for i in self.items_list])

            # Initialize Neural Network Architecture securely
            self.model = self.UCInitModel(
                max(self.n_users, 1), max(self.n_items, 1), self.k, self.mu, 
                P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
            ).to(self.device)
            
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        
    class UCInitModel(nn.Module):
        """Neural embedding layer framework for Latent Factor optimization."""
        def __init__(self, n_users, n_items, k, mu, P_init, Q_init, b_u_init, b_i_init):
            super().__init__()
            self.P = nn.Embedding(n_users, k)
            self.Q = nn.Embedding(n_items, k)
            self.b_u = nn.Embedding(n_users, 1)
            self.b_i = nn.Embedding(n_items, 1)
            self.mu = mu

            # Inject pre-computed topological structures into the embeddings
            with torch.no_grad():
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
            """Forward pass computing the predicted rating."""
            p_u = self.P(user_idx)
            q_i = self.Q(item_idx)
            b_u = self.b_u(user_idx).squeeze(dim=-1)
            b_i = self.b_i(item_idx).squeeze(dim=-1)
            
            dot_product = (p_u * q_i).sum(dim=1)
            return self.mu + b_u + b_i + dot_product
    
    # -------------------------------------------------------------------------
    # Database Operations (Just-in-Time Connection)
    # -------------------------------------------------------------------------
    def load_user_item_from_db(self):
        """Retrieves user and item identifiers associated with the domain."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT "Id" From "User" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.users = [str(row[0]) for row in cur.fetchall()]
                    
                    cur.execute('SELECT "Id" From "Item" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.items_list = [str(row[0]) for row in cur.fetchall()]
        except Exception:
            self.users = []
            self.items_list = []
    
    def load_item_categories(self):
        """Maps each item to its respective categories for structural analysis."""
        category_name = {}
        self.item_categories = defaultdict(list)
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT * FROM "Category"')
                    for row in cur.fetchall():
                        category_name[row[0]] = row[1]
                    
                    if not self.items_list: 
                        return

                    chunk_size = 1000
                    items_int = []
                    for x in self.items_list:
                        try: 
                            items_int.append(int(x))
                        except ValueError: 
                            pass
                        
                    for i in range(0, len(items_int), chunk_size):
                        chunk = items_int[i:i+chunk_size]
                        if not chunk: continue
                        cur.execute(
                            'SELECT "ItemId", "CategoryId" FROM "ItemCategory" WHERE "ItemId" = ANY(%s)',
                            (chunk,)
                        )
                        for iid, cid in cur.fetchall():
                            if cid in category_name:
                                self.item_categories[str(iid)].append(category_name[cid])
        except Exception:
            pass # Fails silently to prevent background task interruption

    def load_ratings_from_db(self, limit_total=1000000, ratio=1.0):
        """Retrieves explicit ratings from the database."""
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
        except Exception:
            rows = []

        if not rows:
            self.ratings = []
            self.test_ratings = []
            return

        total = len(rows)
        train_count = int(total * ratio)
        if train_count == 0 and total > 0:
            train_count = total

        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]

    def load_model_from_db(self, model_id):
        """Loads structural weights and parameters for inference."""
        item_factors, item_biases = {}, {}
        user_factors, user_biases = {}, {}
        temp_k_from_factors = None
        mu_val = 3.5

        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT "ItemId", "ItemBias", "ItemFactors" FROM "ItemFactor" WHERE "ModelId" = %s', (model_id,))
                    for item_id, bias, factors in cur.fetchall():
                        item_id = str(item_id)
                        if item_id in self.item2idx and bias is not None and factors is not None:
                            item_biases[item_id] = float(bias)
                            item_factors[item_id] = np.array(factors, dtype=float)
                            if temp_k_from_factors is None:
                                temp_k_from_factors = len(item_factors[item_id])

                    cur.execute('SELECT "UserId", "UserBias", "UserFactors" FROM "UserFactor" WHERE "ModelId" = %s', (model_id,))
                    for user_id, bias, factors in cur.fetchall():
                        user_id = str(user_id)
                        if user_id in self.user2idx and bias is not None and factors is not None:
                            user_biases[user_id] = float(bias)
                            user_factors[user_id] = np.array(factors, dtype=float)
                            if self.k is None and len(user_factors[user_id]) > 0:
                                self.k = len(user_factors[user_id])

                    cur.execute('SELECT "AverageRating" FROM "Model" WHERE "Id"=%s', (model_id,))
                    mu_row = cur.fetchone()
                    if mu_row and mu_row[0] is not None:
                        mu_val = float(mu_row[0])
        except Exception as e:
            raise RuntimeError(f"Failed to load model state from database: {e}")

        if temp_k_from_factors is not None:
            self.k = temp_k_from_factors

        self.mu = mu_val
        n_users_loaded, n_items_loaded = len(self.users), len(self.items_list)
        
        P_init_arr = np.zeros((n_users_loaded, self.k), dtype=float)
        Q_init_arr = np.zeros((n_items_loaded, self.k), dtype=float)
        b_u_init_arr = np.zeros(n_users_loaded, dtype=float)
        b_i_init_arr = np.zeros(n_items_loaded, dtype=float)
        
        for user_id, idx in self.user2idx.items():
            if user_id in user_factors: P_init_arr[idx] = user_factors[user_id]
            if user_id in user_biases: b_u_init_arr[idx] = user_biases[user_id]

        for item_id, idx in self.item2idx.items():
            if item_id in item_factors: Q_init_arr[idx] = item_factors[item_id]
            if item_id in item_biases: b_i_init_arr[idx] = item_biases[item_id]
                
        self.model = self.UCInitModel(
            max(n_users_loaded, 1), max(n_items_loaded, 1), self.k, self.mu,
            P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
        ).to(self.device)
        self.model.eval()

    def write_model_to_db(self):
        """Persists trained vectors to the database using Batch UPSERT."""
        if self.train_mode != 'train' or self.model is None or self.db_config is None: 
            return

        self.model.eval()

        with torch.no_grad():
            all_user_biases = self.model.b_u.weight.squeeze().cpu().numpy()
            all_user_factors = self.model.P.weight.cpu().numpy()
            all_item_biases = self.model.b_i.weight.squeeze().cpu().numpy()
            all_item_factors = self.model.Q.weight.cpu().numpy()

        user_factor_data, item_factor_data = [], []
        users_in_train = set(u for u, _, _ in self.ratings) if self.ratings else set()
        items_in_train = set(i for _, i, _ in self.ratings) if self.ratings else set()

        for user_id in self.users:
            idx = self.user2idx.get(user_id)
            if user_id in users_in_train and idx is not None:
                user_factor_data.append((user_id, float(all_user_biases[idx]), all_user_factors[idx].tolist(), self.model_id))
            elif idx is not None:
                user_factor_data.append((user_id, 0.0, [0.0]*self.k, self.model_id))
        
        for item_id in self.items_list:
            idx = self.item2idx.get(item_id)
            if item_id in items_in_train and idx is not None:
                item_factor_data.append((item_id, float(all_item_biases[idx]), all_item_factors[idx].tolist(), self.model_id))
            elif idx is not None:
                item_factor_data.append((item_id, 0.0, [0.0]*self.k, self.model_id))

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
                        
                    cur.execute('UPDATE "Model" SET "AverageRating"=%s WHERE "Id"=%s', (float(self.mu), self.model_id))
                conn.commit()
        except Exception:
            pass # Fails silently to prevent background task interruption

    # -------------------------------------------------------------------------
    # UCInit Core Algorithm
    # -------------------------------------------------------------------------
    def _get_top_p_users(self):
        """Identifies the top 'p' most active users based on rating frequency."""
        if not self.ratings: return []
        self.p = min(self.p, self.n_users)
        user_count = Counter([u for u, _, _ in self.ratings])
        top_users = [u for u, _ in user_count.most_common(self.p)]
        return sorted(top_users)

    def _get_top_v_users(self):
        """Identifies users with significant category variance using the Chi-Square test."""
        if not self.ratings: return []
        self.v = min(self.v, self.n_users)

        # Convert categories to a standardized string representation
        try:
            item_to_cat_string = {item: ",".join(self.item_categories[item]) for item in self.items_list}
        except TypeError:
            item_to_cat_string = {
                item: ",".join(self.item_categories.get(item, [])) if self.item_categories.get(item) else "" 
                for item in self.items_list
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
        for item in self.items_list:
            if item in item_to_cat_idx:
                category_totals[item_to_cat_idx[item]] += 1

        user_rated_counts = np.zeros((self.n_users, n_categories), dtype=int)
        for user, item in self.ratings_dict.keys():
            if item in item_to_cat_idx:
                try:
                    user_rated_counts[self.user2idx[user], item_to_cat_idx[item]] += 1
                except KeyError:
                    pass 

        user_scores = {}
        contingency_table = np.zeros((2, n_categories), dtype=int) 

        for u_idx, user in enumerate(self.users):
            rated_row = user_rated_counts[u_idx, :]
            not_rated_row = category_totals - rated_row

            contingency_table[0, :] = rated_row
            contingency_table[1, :] = not_rated_row
            
            # Perform Chi-Square test to evaluate rating distribution independence
            if contingency_table.sum() == 0 or np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
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
        """Constructs P and Q matrices initialized by top diverse users."""
        top_v_idx = [self.user2idx[u] for u in self.top_v_users if u in self.user2idx]
        R = lil_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        for u, i, r in self.ratings:
            if u in self.user2idx and i in self.item2idx:
                R[self.user2idx[u], self.item2idx[i]] = r

        R_top = R[top_v_idx, :].toarray()
        Q_init_matrix = R_top.T

        # Pad matrix horizontally if structural dimension 'v' is less than latent dimension 'k'
        if Q_init_matrix.shape[1] < self.k:
            padding = np.zeros((Q_init_matrix.shape[0], self.k - Q_init_matrix.shape[1]), dtype=np.float32)
            Q_init_matrix = np.hstack([Q_init_matrix, padding])
        else:
            Q_init_matrix = Q_init_matrix[:, :self.k]

        Q_map = {}
        for idx, item in enumerate(self.items_list):
            vec = Q_init_matrix[idx]
            norm = np.linalg.norm(vec)
            Q_map[item] = vec / norm if norm > 0 else vec

        # Compute P mappings via normalized projection
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
        """Wraps matrices and biases initialization."""
        P, Q = self._UCInit()
        b_u = {user: 0.0 for user in self.users}
        b_i = {item: 0.0 for item in self.items_list}
        return P, Q, b_u, b_i

    # -------------------------------------------------------------------------
    # Training and Inference
    # -------------------------------------------------------------------------
    def train_model(self, epochs=500, batch_size=256):
        """Execute the optimization loop using Mini-Batch Gradient Descent."""
        if not self.ratings or self.model is None:
            return
        
        self.model.train()
        n = len(self.ratings)
        tolerance = 1e-6
        prev_avg_loss = float("inf")

        # Explicitly declare and convert to tensors to match exactly
        u_all = torch.tensor([self.user2idx[u] for u, _, _ in self.ratings], device=self.device, dtype=torch.long)
        i_all = torch.tensor([self.item2idx[i] for _, i, _ in self.ratings], device=self.device, dtype=torch.long)
        r_all = torch.tensor([r for _, _, r in self.ratings], device=self.device, dtype=torch.float32)

        for _ in range(epochs):
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
                main_loss_mean = (0.5 * (err_r**2)).mean()
                
                self.optimizer.zero_grad()
                main_loss_mean.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_mse_sum += main_loss_mean.item() * len(r_batch)

            avg_mse_loss = total_mse_sum / n
            
            with torch.no_grad():
                reg_p = torch.sum(self.model.P.weight**2)
                reg_q = torch.sum(self.model.Q.weight**2)
                reg_bu = torch.sum(self.model.b_u.weight**2)
                reg_bi = torch.sum(self.model.b_i.weight**2)
            
            final_reg_loss_total_sum = 0.5 * self.lam * (reg_p + reg_q + reg_bu + reg_bi).item()
            avg_total_loss = avg_mse_loss + final_reg_loss_total_sum / n
            
            if abs(prev_avg_loss - avg_total_loss) < tolerance:
                break
            prev_avg_loss = avg_total_loss

    def predict(self, user, item, p=0):
        """Infers the rating score for a specific user-item pair."""
        if self.model is None:
            return self.mu if self.mu else 3.5

        self.model.eval()
        with torch.no_grad():
            u_idx_val = self.user2idx.get(user)
            i_idx_val = self.item2idx.get(item)
            
            # Handle cold-start scenarios for unobserved users or items
            if u_idx_val is None:
                if i_idx_val is not None:
                    i_idx = torch.tensor([i_idx_val], device=self.device, dtype=torch.long)
                    prediction = self.mu + self.model.b_i(i_idx).squeeze().item()
                else:
                    prediction = self.mu
                return np.clip(prediction, 1.0, 5.0)

            if i_idx_val is None:
                u_idx = torch.tensor([u_idx_val], device=self.device, dtype=torch.long)
                prediction = self.mu + self.model.b_u(u_idx).squeeze().item()
                return np.clip(prediction, 1.0, 5.0)

            u_idx = torch.tensor([u_idx_val], device=self.device, dtype=torch.long)
            i_idx = torch.tensor([i_idx_val], device=self.device, dtype=torch.long)
            prediction = self.model(u_idx, i_idx).item()
            
            return np.clip(prediction, 1.0, 5.0)