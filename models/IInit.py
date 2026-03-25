import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import NMF
import psycopg2
from psycopg2 import extras
from scipy.sparse import lil_matrix

class LatentFactorModel(nn.Module):
    """
    Latent Factor Model for Implicit Feedback (IInit).
    Uses Non-Negative Matrix Factorization (NMF) to initialize weights,
    followed by fine-tuning using PyTorch Embedding layers.
    """
    def __init__(
        self,
        db_config=None,
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
        self.db_config = db_config
        
        if self.db_config is None:
            raise ValueError("DB config is required to connect to the database.")
        if train_mode not in ['train', 'load']:
            raise ValueError("train_mode must be 'train' or 'load'.")
        
        self.weight = weight
        self.lam = lam
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.k = k
        self.domain_id = domain_id
        self.interaction_type_id = interaction_type_id
        
        self.model = None
        self.optimizer = None
        self.users = []
        self.items_list = []
        self.user2idx = {}
        self.item2idx = {}
        self.ratings = []
        self.mu = 3.5

        # Initialize data and model based on the selected mode (train/load)
        self._initialize_system()

    def _initialize_system(self):
        """Internal method to manage data initialization and model weight loading."""
        if self.train_mode == 'load':
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            self.load_model_from_db(self.model_id)
            
        else: # train_mode == 'train'
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            self.n_users = len(self.users)
            self.n_items = len(self.items_list)

            self.load_ratings_from_db()

            # Handle Cold Start (Missing Data)
            if not self.ratings:
                self.mu = 3.5
                P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                b_u_init_arr = np.zeros(self.n_users)
                b_i_init_arr = np.zeros(self.n_items)
            else:
                if self.interaction_type_id == 0:
                    raise ValueError("interaction_type_id must be greater than 0.")
                
                # Filter out invalid user/item interactions
                self.ratings = [(u, i, r) for u, i, r in self.ratings if u in self.user2idx and i in self.item2idx]

                if not self.ratings:
                     self.mu = 3.5
                     P_init_arr = np.random.normal(scale=0.01, size=(self.n_users, self.k))
                     Q_init_arr = np.random.normal(scale=0.01, size=(self.n_items, self.k))
                     b_u_init_arr = np.zeros(self.n_users)
                     b_i_init_arr = np.zeros(self.n_items)
                else:
                    # Standard Flow: Build interaction matrix -> NMF -> Init Model
                    self.mu = np.mean([r for _, _, r in self.ratings])
                    self.build_interaction_matrix()
                    P_map, Q_map, b_u_map, b_i_map = self.init_latent_model()
                    
                    P_init_arr = P_map
                    Q_init_arr = Q_map.T
                    b_u_init_arr = np.array([b_u_map[u] for u in self.users])
                    b_i_init_arr = np.array([b_i_map[i] for i in self.items_list])

            # Initialize PyTorch Model architecture
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
        """Neural embedding layer for Users and Items."""
        def __init__(self, n_users, n_items, k, mu, P_init, Q_init, b_u_init, b_i_init):
            super().__init__()
            self.P = nn.Embedding(n_users, k)
            self.Q = nn.Embedding(n_items, k)
            self.b_u = nn.Embedding(n_users, 1)
            self.b_i = nn.Embedding(n_items, 1)
            self.mu = mu
            
            # Load initial weights (from NMF or Random)
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
            p_u = self.P(user_idx)
            q_i = self.Q(item_idx)
            b_u = self.b_u(user_idx).squeeze(dim=-1)
            b_i = self.b_i(item_idx).squeeze(dim=-1)
            
            dot_product = (p_u * q_i).sum(dim=1)
            return self.mu + b_u + b_i + dot_product
    
    # ---------------------------------------------------------
    # Database Operations (Just-in-Time Connection)
    # ---------------------------------------------------------
    def load_user_item_from_db(self):
        """Load the complete list of Users and Items for the current Domain."""
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
    
    def load_model_from_db(self, model_id):
        """Load previously trained model weights from the database."""
        item_factors, item_biases = {}, {}
        user_factors, user_biases = {}, {}
        temp_k_from_factors = None
        mu_val = 3.5

        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Load Item Factors
                cur.execute('SELECT "ItemId", "ItemBias", "ItemFactors" FROM "ItemFactor" WHERE "ModelId" = %s', (model_id,))
                for item_id, bias, factors in cur.fetchall():
                    item_id = str(item_id)
                    if item_id in self.item2idx and bias is not None and factors is not None:
                        item_biases[item_id] = float(bias)
                        item_factors[item_id] = np.array(factors, dtype=float)
                        if temp_k_from_factors is None:
                            temp_k_from_factors = len(item_factors[item_id])

                # Load User Factors
                cur.execute('SELECT "UserId", "UserBias", "UserFactors" FROM "UserFactor" WHERE "ModelId" = %s', (model_id,))
                for user_id, bias, factors in cur.fetchall():
                    user_id = str(user_id)
                    if user_id in self.user2idx and bias is not None and factors is not None:
                        user_biases[user_id] = float(bias)
                        user_factors[user_id] = np.array(factors, dtype=float)
                        if self.k is None and len(user_factors[user_id]) > 0:
                            self.k = len(user_factors[user_id])

                # Load Average Rating (Mu)
                cur.execute('SELECT "AverageRating" FROM "Model" WHERE "Id"=%s', (model_id,))
                mu_row = cur.fetchone()
                if mu_row and mu_row[0] is not None:
                    mu_val = float(mu_row[0])
                        
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
                
        self.model = self.IInitModel(
            max(n_users_loaded, 1), max(n_items_loaded, 1), self.k, self.mu,
            P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
        ).to(self.device)
        self.model.eval()
    
    def load_ratings_from_db(self, limit_total=1000000, ratio=1.0):
        """Load interaction data to create implicit feedback."""
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
        except Exception:
            rows = []
        
        if not rows:
            self.ratings = []
            return

        total = len(rows)
        train_count = int(total * ratio)
        if train_count == 0 and total > 0: 
            train_count = total
        
        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]

    def write_model_to_db(self):
        """Save the entire P, Q matrices and Biases to the database."""
        if self.train_mode != 'train' or self.model is None: 
            return

        self.model.eval()
        with torch.no_grad():
            all_user_biases = self.model.b_u.weight.squeeze().cpu().numpy()
            all_user_factors = self.model.P.weight.cpu().numpy()
            all_item_biases = self.model.b_i.weight.squeeze().cpu().numpy()
            all_item_factors = self.model.Q.weight.cpu().numpy()

        user_factor_data, item_factor_data = [], []
        users_in_train = set(u for u, _, _ in self.ratings)
        items_in_train = set(i for _, i, _ in self.ratings)

        for user_id in self.users:
            idx = self.user2idx.get(user_id)
            if user_id in users_in_train and idx is not None:
                user_factor_data.append((user_id, float(all_user_biases[idx]), all_user_factors[idx].tolist(), self.model_id))
            else:
                user_factor_data.append((user_id, 0.0, [0.0]*self.k, self.model_id))
        
        for item_id in self.items_list:
            idx = self.item2idx.get(item_id)
            if item_id in items_in_train and idx is not None:
                item_factor_data.append((item_id, float(all_item_biases[idx]), all_item_factors[idx].tolist(), self.model_id))
            else:
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
            pass # Silent fail during background processing to prevent thread crash

    # ---------------------------------------------------------
    # NMF Initialization and Training Loop
    # ---------------------------------------------------------
    def build_interaction_matrix(self):
        """Build a sparse User-Item matrix from interaction history."""
        interaction_matrix = lil_matrix((self.n_users, self.n_items), dtype=np.int8)
        for u, i, _ in self.ratings:
             if u in self.user2idx and i in self.item2idx:
                user_idx = self.user2idx[u]
                item_idx = self.item2idx[i]
                interaction_matrix[user_idx, item_idx] = 1
                
        self.interaction_matrix = interaction_matrix
        
    def IInit(self):
        """Factorize the interaction matrix (NMF) to extract base feature vectors."""
        if self.interaction_matrix.sum() == 0:
            return np.random.rand(self.n_users, self.k), np.random.rand(self.k, self.n_items)
            
        model = NMF(n_components=self.k, init="random", random_state=42)
        P = model.fit_transform(self.interaction_matrix)
        Q = model.components_
        return P, Q

    def init_latent_model(self):
        """Integrate NMF results into standard format."""
        P, Q = self.IInit()
        b_u = {u: 0.0 for u in self.users}
        b_i = {i: 0.0 for i in self.items_list}
        return P, Q, b_u, b_i
    
    def train_model(self, epochs=500, batch_size=256):
        """Training loop to optimize Embeddings using PyTorch."""
        if not self.ratings or self.model is None:
            return
        
        self.model.train()
        
        u_all = torch.tensor([self.user2idx[u] for u, _, _ in self.ratings], device=self.device, dtype=torch.long)
        i_all = torch.tensor([self.item2idx[i] for _, i, _ in self.ratings], device=self.device, dtype=torch.long)
        r_all = torch.tensor([r for _, _, r in self.ratings], device=self.device, dtype=torch.float32)
        
        tol = 1e-6
        n = len(r_all)
        prev_avg_loss = float("inf")

        for epoch_num in range(epochs):
            total_mse_sum = 0.0 
            
            # Shuffle data
            perm = torch.randperm(n, device=self.device)
            u_shuf = u_all[perm]
            i_shuf = i_all[perm]
            r_shuf = r_all[perm]

            # Mini-batch loop
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                if start == end: continue
                
                u_batch = u_shuf[start:end] 
                i_batch = i_shuf[start:end]
                r_batch = r_shuf[start:end]
                
                # Predict and calculate Loss
                r_hat = self.model(u_batch, i_batch)
                err_r = r_batch - r_hat
                main_loss_mean = (0.5 * (err_r**2)).mean()
                
                # Backpropagation
                self.optimizer.zero_grad()
                main_loss_mean.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_mse_sum += main_loss_mean.item() * len(r_batch)

            # Calculate Regularization Loss and check Early Stopping
            avg_mse_loss = total_mse_sum / n
            with torch.no_grad():
                reg_p = torch.sum(self.model.P.weight**2)
                reg_q = torch.sum(self.model.Q.weight**2)
                reg_bu = torch.sum(self.model.b_u.weight**2)
                reg_bi = torch.sum(self.model.b_i.weight**2)
            
            final_reg_loss = 0.5 * self.lam * (reg_p + reg_q + reg_bu + reg_bi).item()
            avg_total_loss = avg_mse_loss + final_reg_loss / n
            
            if abs(prev_avg_loss - avg_total_loss) < tol:
                break
            prev_avg_loss = avg_total_loss

    def predict(self, user, item, p=0):
        """Inference function to predict the score for a single User-Item pair."""
        if self.model is None:
            return self.mu if self.mu else 3.0

        self.model.eval()
        with torch.no_grad():
            u_idx_val = self.user2idx.get(user)
            i_idx_val = self.item2idx.get(item)
            
            # Handle cases where User or Item was unseen during training
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

            # Both User and Item are valid
            u_idx = torch.tensor([u_idx_val], device=self.device, dtype=torch.long)
            i_idx = torch.tensor([i_idx_val], device=self.device, dtype=torch.long)
            prediction = self.model(u_idx, i_idx).item()
            
            return np.clip(prediction, 1.0, 5.0)