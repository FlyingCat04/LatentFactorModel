from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import psycopg2
from psycopg2 import extras
from tqdm import tqdm

class LatentFactorModel(nn.Module):
    """
    User-Item Embedding Interactive (UEIE) Model.
    
    This model utilizes DistilBERT to extract textual features from items, aggregates 
    them to infer user preferences, and fine-tunes the embeddings using a PyTorch 
    Latent Factor Model approach.
    """
    def __init__(
        self,
        db_config=None,
        ratings=None, 
        items=None,
        k=90,
        weight=0.3,
        lam=0.001,
        lr=0.01,
        model_id=1,
        domain_id=None,
        train_mode='train',
        device=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.db_config = db_config
        self.weight = weight
        self.lam = lam
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.k = k
        self.domain_id = domain_id
        
        # NLP and Embedding variables
        self.tokenizer = None
        self.bert_model = None
        self.user_emb_dict = {}
        self.item_emb_dict = {}
        
        # System variables
        self.ratings = [] 
        self.test_ratings = []
        self.users = []
        self.items_list = []
        self.user2idx = {}
        self.item2idx = {}
        self.mu = 3.5
        
        self.model = None
        self.optimizer = None
        self.inferred_prefs = None

        if self.db_config is None:
            raise ValueError("Database configuration is required.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("train_mode must be strictly 'train' or 'load'.")

        # Delegate initialization logic to keep constructor clean
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

            if input_ratings is None:
                self.load_ratings_from_db(ratio=1.0)
            else:
                self.ratings = input_ratings

            # Extract unique users and items from the training set
            users_in_train_set = set(u for u, _, _ in self.ratings) if self.ratings else set()
            items_in_train_set = set(i for _, i, _ in self.ratings) if self.ratings else set()
            users_in_train = list(users_in_train_set)
            items_in_train = list(items_in_train_set)

            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: j for j, i in enumerate(self.items_list)}
            
            # Retrieve or generate BERT embeddings
            if self.users and self.items_list:
                self.load_user_item_embeddings_from_db(self.users, self.items_list)
            
            # Filter valid ratings
            if self.ratings:
                self.ratings = [(u, i, r) for u, i, r in self.ratings if u in self.user2idx and i in self.item2idx]
            
            self.compute_user_embeddings()

            self.mu = np.mean([r for _, _, r in self.ratings]) if self.ratings else 3.5

            n_users_final = len(self.users)
            n_items_final = len(self.items_list)
            
            self.model = self.UEIEModel(n_users_final, n_items_final, self.k, self.mu).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
            
            self.inferred_prefs = self.compute_inferred_preferences()

    class UEIEModel(nn.Module):
        """Neural Embedding Architecture for the Latent Factor Model."""
        def __init__(self, n_users, n_items, k, mu, P_init=None, Q_init=None, b_u_init=None, b_i_init=None):
            super().__init__()
            # Ensure dimensions are strictly positive to prevent Embedding layer crashes
            n_users = max(n_users, 1)
            n_items = max(n_items, 1)

            self.P = nn.Embedding(n_users, k)
            self.Q = nn.Embedding(n_items, k)
            self.b_u = nn.Embedding(n_users, 1)
            self.b_i = nn.Embedding(n_items, 1)
            self.mu = mu
            
            # Initialize weights safely (copying pre-computed tensors if dimensions match)
            with torch.no_grad():
                if P_init is not None and P_init.shape == self.P.weight.shape:
                    self.P.weight.copy_(torch.tensor(P_init, dtype=torch.float32))
                else:
                    nn.init.xavier_uniform_(self.P.weight)

                if Q_init is not None and Q_init.shape == self.Q.weight.shape:
                    self.Q.weight.copy_(torch.tensor(Q_init, dtype=torch.float32))
                else:
                    nn.init.xavier_uniform_(self.Q.weight)

                if b_u_init is not None and b_u_init.shape == self.b_u.weight.squeeze().shape:
                    self.b_u.weight.copy_(torch.tensor(b_u_init, dtype=torch.float32).unsqueeze(1))
                else:
                    nn.init.zeros_(self.b_u.weight)

                if b_i_init is not None and b_i_init.shape == self.b_i.weight.squeeze().shape:
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
    # Core Logic: Embeddings & Inferred Preferences
    # -------------------------------------------------------------------------
    def compute_user_embeddings(self):
        """Constructs user embeddings by averaging the embeddings of items they positively interacted with."""
        if not self.item_emb_dict:
            k_dim_bert = self.k
            for user_id in self.users:
                if user_id not in self.user_emb_dict:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
            return

        k_dim_bert = len(next(iter(self.item_emb_dict.values()))) if self.item_emb_dict else 768

        # Map users to their respective ratings
        user_ratings_map = defaultdict(list)
        for u, i, r in self.ratings:
            if u in self.user2idx:
                user_ratings_map[u].append((i, r))

        for user_id in self.users:
            ratings_list = user_ratings_map.get(user_id, [])
            if not ratings_list:
                if user_id not in self.user_emb_dict or self.user_emb_dict[user_id] is None:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
                continue

            # Define threshold tau as the user's average rating
            tau = np.mean([r for _, r in ratings_list])
            liked_item_embeddings = []
            
            for item_id, rating in ratings_list:
                if rating >= tau:
                    if item_id in self.item_emb_dict and np.any(self.item_emb_dict[item_id] != 0):
                        liked_item_embeddings.append(self.item_emb_dict[item_id])

            # Aggregate embeddings
            if liked_item_embeddings:
                t_u = np.mean(liked_item_embeddings, axis=0)
                self.user_emb_dict[user_id] = t_u if t_u.shape[0] == k_dim_bert else np.zeros(k_dim_bert)
            else:
                # Fallback: compute mean of all rated items if none exceed the threshold
                all_rated_items = [self.item_emb_dict[i] for i, _ in ratings_list if i in self.item_emb_dict and np.any(self.item_emb_dict[i] != 0)]
                self.user_emb_dict[user_id] = np.mean(all_rated_items, axis=0) if all_rated_items else np.zeros(k_dim_bert)
                
    def compute_inferred_preferences(self):
        """Calculates and normalizes the dot product between User and Item text embeddings (z_ui)."""
        if not self.ratings or not self.user_emb_dict or not self.item_emb_dict:
            return np.zeros(len(self.ratings) if self.ratings else 0)

        inferred = np.zeros(len(self.ratings))
        try:
            k_dim_check = len(next(iter(self.item_emb_dict.values())))
        except StopIteration:
            return np.zeros(len(self.ratings))

        for idx, (u, i, _) in enumerate(self.ratings):
            user_vec = self.user_emb_dict.get(u)
            item_vec = self.item_emb_dict.get(i)

            valid = True
            if user_vec is None or user_vec.shape[0] != k_dim_check:
                valid = False
            if item_vec is None or item_vec.shape[0] != k_dim_check:
                valid = False

            if valid:
                try:
                    inferred[idx] = np.dot(user_vec, item_vec)
                except ValueError:
                    inferred[idx] = 0.0
            else:
                inferred[idx] = 0.0

        # Handle potential numerical instability
        if np.isnan(inferred).any() or np.isinf(inferred).any():
            inferred = np.nan_to_num(inferred, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale inferred preferences to align with the rating scale [1, 5]
        if np.ptp(inferred) > 1e-9:
            scaler = MinMaxScaler(feature_range=(1, 5))
            try:
                return scaler.fit_transform(inferred.reshape(-1, 1)).flatten()
            except ValueError:
                return np.clip(inferred, 1.0, 5.0)
        else:
            return np.full(len(self.ratings), self.mu if self.mu != 0.0 else 3.5)

    # -------------------------------------------------------------------------
    # BERT Operations
    # -------------------------------------------------------------------------
    def load_bert(self):
        """Lazy instantiation of the DistilBERT components."""
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.bert_model.eval()
            self.bert_model.to(self.device)
            return self.tokenizer, self.bert_model
        except Exception as e:
            self.tokenizer, self.bert_model = None, None
            raise RuntimeError(f"Failed to load DistilBERT model: {e}")

    def get_embedding(self, text):
        """Generates contextual embeddings for a given text segment."""
        if self.tokenizer is None or self.bert_model is None:
            return np.zeros(768)

        if not text or not isinstance(text, str):
            return np.zeros(self.bert_model.config.dim)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = self.bert_model(**inputs)
            except Exception:
                return np.zeros(self.bert_model.config.dim)

        # Mean pooling to extract sentence-level representation
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.squeeze().cpu().numpy()

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

        train_count = int(len(rows) * ratio)
        if train_count == 0 and len(rows) > 0: 
            train_count = len(rows)

        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]

    def load_user_item_embeddings_from_db(self, users_to_load, items_to_load):
        """Retrieves saved text embeddings or generates them via BERT if unavailable."""
        if not users_to_load or not items_to_load:
            return

        CHUNK_SIZE = 1000
        k_bert_dim = 768
        all_rows_items = []
        items_list_for_query = list(items_to_load)
        
        # Phase 1: Retrieve item embeddings
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    for i in range(0, len(items_list_for_query), CHUNK_SIZE):
                        chunk = items_list_for_query[i : i + CHUNK_SIZE]
                        if not chunk: continue
                        cur.execute("""
                            SELECT "Id", 
                                   COALESCE(NULLIF("Description", ''), "Title") AS "Description", 
                                   "EmbeddingVector"
                            FROM "Item"
                            WHERE "DomainId" = %s AND "Id" = ANY(%s::int[])
                        """, (self.domain_id, chunk,))
                        all_rows_items.extend(cur.fetchall())
        except Exception as e:
            raise RuntimeError(f"Error querying item chunks: {e}")

        self.item_emb_dict = {}
        missing_items_info = []
        processed_item_ids = set()
        
        for item_id, desc, emb_vec in all_rows_items:
            item_id = str(item_id)
            if emb_vec is not None:
                try:
                    loaded_emb = np.array(emb_vec, dtype=float)
                    if len(loaded_emb) > 0 and np.any(loaded_emb != 0):
                        self.item_emb_dict[item_id] = loaded_emb
                        k_bert_dim = len(loaded_emb) 
                    else:
                        missing_items_info.append((item_id, desc))
                except Exception:
                    missing_items_info.append((item_id, desc))
            else:
                if desc is not None or desc != "":
                    missing_items_info.append((item_id, desc))
            processed_item_ids.add(item_id)

        items_not_in_db_table = set(items_to_load) - processed_item_ids
        for item_id in items_not_in_db_table:
            missing_items_info.append((item_id, None))

        # Phase 2: Compute missing embeddings locally via DistilBERT
        if missing_items_info:
            if self.tokenizer is None or self.bert_model is None:
                try:
                    self.tokenizer, self.bert_model = self.load_bert()
                    if hasattr(self.bert_model, 'config') and hasattr(self.bert_model.config, 'dim'):
                        k_bert_dim = self.bert_model.config.dim
                except Exception:
                    pass
            
            # Use tqdm silently for internal processing progress
            for item_id, desc in tqdm(missing_items_info, desc="Computing BERT embeddings", disable=True):
                if item_id not in self.item_emb_dict:
                    if self.bert_model:
                        bert_emb = self.get_embedding(desc)
                        self.item_emb_dict[item_id] = bert_emb if bert_emb.shape[0] == k_bert_dim else np.zeros(k_bert_dim)
                    else:
                        self.item_emb_dict[item_id] = np.zeros(k_bert_dim)
        
        # Phase 3: Retrieve user embeddings
        all_rows_users = []
        users_list_for_query = list(users_to_load)
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    for i in range(0, len(users_list_for_query), CHUNK_SIZE):
                        chunk = users_list_for_query[i : i + CHUNK_SIZE]
                        if not chunk: continue
                        cur.execute("""
                            SELECT "Id", "UserEmbeddingVector"
                            FROM "User"
                            WHERE "DomainId" = %s AND "Id" = ANY(%s::int[])
                        """, (self.domain_id, chunk,))
                        all_rows_users.extend(cur.fetchall())
        except Exception as e:
            raise RuntimeError(f"Error querying user chunks: {e}")

        self.user_emb_dict = {}
        processed_user_ids = set()
        k_dim_for_user = k_bert_dim
        
        for user_id, emb_vec in all_rows_users:
            user_id = str(user_id)
            if emb_vec is not None:
                try:
                    loaded_emb = np.array(emb_vec, dtype=float)
                    self.user_emb_dict[user_id] = loaded_emb if len(loaded_emb) == k_dim_for_user else np.zeros(k_dim_for_user)
                except Exception:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
            else:
                self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
            processed_user_ids.add(user_id)

        users_not_in_db_table = set(users_to_load) - processed_user_ids
        for user_id in users_not_in_db_table:
            self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)

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
                
        self.model = self.UEIEModel(n_users_loaded, n_items_loaded, self.k, self.mu, P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr).to(self.device)
        self.model.eval()

    def write_model_to_db(self):
        """Persists trained vectors and NLP embeddings to the database."""
        if self.train_mode != 'train' or self.model is None or self.db_config is None: 
            return
        
        self.model.eval()
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Update static NLP embeddings
                    if self.user_emb_dict and self.item_emb_dict:
                        try:
                            for user_id in self.users:
                                if user_id in self.user_emb_dict:
                                    emb = self.user_emb_dict[user_id]
                                    cur.execute('UPDATE "User" SET "UserEmbeddingVector"=%s, "ModifiedAt"=NOW() WHERE "Id"=%s', (emb.tolist(), user_id))
                            for item_id in self.items_list:
                                if item_id in self.item_emb_dict:
                                    emb = self.item_emb_dict[item_id]
                                    cur.execute('UPDATE "Item" SET "EmbeddingVector"=%s, "ModifiedAt"=NOW() WHERE "Id"=%s', (emb.tolist(), item_id))
                            conn.commit()
                        except Exception:
                            conn.rollback()

                    # Extract Latent Factors from PyTorch tensors
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
                            user_factor_data.append((user_id, 0.0, all_user_factors[idx].tolist(), self.model_id))
                    
                    for item_id in self.items_list:
                        idx = self.item2idx.get(item_id)
                        if item_id in items_in_train and idx is not None:
                            item_factor_data.append((item_id, float(all_item_biases[idx]), all_item_factors[idx].tolist(), self.model_id))
                        elif idx is not None:
                            item_factor_data.append((item_id, 0.0, all_item_factors[idx].tolist(), self.model_id))

                    # Batch UPSERT configuration
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
                    
                    cur.execute('UPDATE "Model" SET "AverageRating"=%s, "ModifiedAt"=NOW() WHERE "Id"=%s', (float(self.mu), self.model_id))
                    conn.commit()

        except Exception:
            pass # Fails silently to prevent blocking the worker queue

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
        prev_avg_loss = float('inf')

        user_indices = [self.user2idx[u] for u, _, _ in self.ratings]
        item_indices = [self.item2idx[i] for _, i, _ in self.ratings]
        user_idx_all = torch.tensor(user_indices, device=self.device, dtype=torch.long)
        item_idx_all = torch.tensor(item_indices, device=self.device, dtype=torch.long)
        r_all = torch.tensor([r for _, _, r in self.ratings], device=self.device, dtype=torch.float32)

        if self.inferred_prefs is None or len(self.inferred_prefs) != n:
            z_all = torch.full((n,), self.mu, dtype=torch.float32, device=self.device)
        else:
            z_all = torch.tensor(self.inferred_prefs, dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            total_main_loss_sum = 0.0
            
            # Stochastic shuffling
            perm = torch.randperm(n, device=self.device)
            user_idx_shuffled = user_idx_all[perm]
            item_idx_shuffled = item_idx_all[perm]
            r_shuffled = r_all[perm]
            z_shuffled = z_all[perm]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                if start == end: continue
                
                u_batch = user_idx_shuffled[start:end]
                i_batch = item_idx_shuffled[start:end]
                r_batch = r_shuffled[start:end]
                z_batch = z_shuffled[start:end]
                
                r_hat = self.model(u_batch, i_batch)
                
                # Hybrid loss: explicit rating error + weighted implicit preference error
                err_r = r_batch - r_hat
                err_z = z_batch - r_hat
                main_loss_mean = (0.5 * (err_r**2 + self.weight * err_z**2)).mean()

                self.optimizer.zero_grad()
                main_loss_mean.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_main_loss_sum += main_loss_mean.item() * len(u_batch)

            # Evaluate early stopping criteria
            avg_main_loss = total_main_loss_sum / n
            with torch.no_grad():
                reg_p = torch.sum(self.model.P.weight**2)
                reg_q = torch.sum(self.model.Q.weight**2)
                reg_bu = torch.sum(self.model.b_u.weight**2)
                reg_bi = torch.sum(self.model.b_i.weight**2)
            
            final_reg_loss = 0.5 * self.lam * (reg_p + reg_q + reg_bu + reg_bi).item()
            avg_total_loss = avg_main_loss + final_reg_loss / n

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