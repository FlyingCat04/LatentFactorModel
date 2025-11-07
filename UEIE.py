from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class LatentFactorModel(nn.Module):
    def __init__(
        self,
        connection,
        ratings=None, 
        items=None,
        k=90,
        weight=0.3,
        lam=0.001,
        lr=0.01,
        model_id=1,
        train_mode='train',
        device=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.connection = connection
        self.weight = weight
        self.lam = lam
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.k = k
        self.tokenizer = None
        self.bert_model = None
        self.ratings = None
        self.test_ratings = None
        self.user_emb_dict = {}
        self.item_emb_dict = {}
        self.users = []
        self.items_list = []
        self.user2idx = {}
        self.item2idx = {}
        self.mu = 0.0
        self.model = None
        self.optimizer = None
        self.inferred_prefs = None

        if self.connection is None and self.train_mode == 'train' and (ratings is None or items is None):
            raise ValueError("❌ DB connection or ratings/items required for training.")
        if self.connection is None and self.train_mode == 'load':
            raise ValueError("❌ DB connection required for loading model.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("❌ train_mode must be 'train' or 'load'")

        if self.train_mode == 'load':
            print("--- Loading Existing Model ---")
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: j for j, i in enumerate(self.items_list)}
            self.load_model_from_db(model_id=model_id)
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

            self.load_user_item_embeddings_from_db(users_in_train, items_in_train)
            self.load_user_item_from_db()
            
            # self.users = sorted(list(self.user_emb_dict.keys()))
            # self.items_list = sorted(list(self.item_emb_dict.keys()))
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: j for j, i in enumerate(self.items_list)}
            self.compute_user_embeddings()

            self.mu = np.mean([r for _, _, r in self.ratings]) if self.ratings else 3.5

            n_users_final = len(self.users)
            n_items_final = len(self.items_list)
            if n_users_final == 0 or n_items_final == 0:
                raise ValueError(f"❌ Cannot initialize model with {n_users_final} users / {n_items_final} items.")
            self.model = self.UEIEModel(n_users_final, n_items_final, self.k, self.mu).to(self.device)

            self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=self.lr, 
                                   weight_decay=self.lam)
            self.inferred_prefs = self.compute_inferred_preferences()
            print("✅ Initialization complete and ready for training.")

    class UEIEModel(nn.Module):
        def __init__(self, n_users, n_items, k, mu, P_init=None, Q_init=None, b_u_init=None, b_i_init=None):
            super().__init__()
            self.P = nn.Embedding(n_users, k)
            self.Q = nn.Embedding(n_items, k)
            self.b_u = nn.Embedding(n_users, 1)
            self.b_i = nn.Embedding(n_items, 1)
            self.mu = mu
            
            if P_init is not None:
                with torch.no_grad():
                    self.P.weight.copy_(torch.tensor(P_init, dtype=torch.float32))
                    self.Q.weight.copy_(torch.tensor(Q_init, dtype=torch.float32))
                    self.b_u.weight.copy_(torch.tensor(b_u_init, dtype=torch.float32).unsqueeze(1))
                    self.b_i.weight.copy_(torch.tensor(b_i_init, dtype=torch.float32).unsqueeze(1))
            else:
                nn.init.xavier_uniform_(self.P.weight)
                nn.init.xavier_uniform_(self.Q.weight)
                nn.init.zeros_(self.b_u.weight)
                nn.init.zeros_(self.b_i.weight)

        def forward(self, user_idx, item_idx):
            p_u = self.P(user_idx)
            q_i = self.Q(item_idx)
            b_u = self.b_u(user_idx).squeeze(dim=-1)
            b_i = self.b_i(item_idx).squeeze(dim=-1)
            dot_product = (p_u * q_i).sum(dim=1)
            return self.mu + b_u + b_i + dot_product

    def compute_user_embeddings(self):
        if not self.item_emb_dict:
            k_dim_bert = self.k
            for user_id in self.users:
                if user_id not in self.user_emb_dict:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
            return

        k_dim_bert = 768

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

            user_avg_rating = np.mean([r for _, r in ratings_list])
            tau = user_avg_rating

            liked_item_embeddings = []
            for item_id, rating in ratings_list:
                if rating > tau:
                    if item_id in self.item_emb_dict:
                        liked_item_embeddings.append(self.item_emb_dict[item_id])

            if liked_item_embeddings:
                t_u = np.mean(liked_item_embeddings, axis=0)
                if t_u.shape[0] != k_dim_bert:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
                else:
                    self.user_emb_dict[user_id] = t_u
            else:
                self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
                
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

        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]

    def load_user_item_embeddings_from_db(self, users_to_load, items_to_load):
        if not users_to_load or not items_to_load:
            return

        cur = self.connection.cursor()
        CHUNK_SIZE = 1000
        k_bert_dim = 768
        all_rows_items = []
        items_list_for_query = list(items_to_load)
        for i in range(0, len(items_list_for_query), CHUNK_SIZE):
            chunk = items_list_for_query[i : i + CHUNK_SIZE]
            if not chunk: continue
            try:
                cur.execute("""
                    SELECT "ItemID", "Description", "ItemEmbeddingVector"
                    FROM "Item"
                    WHERE "ItemID" = ANY(%s::int[])
                """, (chunk,))
                rows_chunk = cur.fetchall()
                all_rows_items.extend(rows_chunk)
            except Exception as e:
                print(f"❌ Error querying item chunk starting at index {i}: {e}")
                raise

        self.item_emb_dict = {}
        missing_items_info = []
        processed_item_ids = set()
        for item_id, desc, emb_vec in all_rows_items:
            if emb_vec is not None:
                try:
                    loaded_emb = np.array(emb_vec, dtype=float)
                    if len(loaded_emb) == k_bert_dim:
                        self.item_emb_dict[item_id] = loaded_emb
                    else:
                        missing_items_info.append((item_id, desc))
                except Exception as e:
                    missing_items_info.append((item_id, desc))
            else:
                if desc is not None or desc != "":
                    missing_items_info.append((item_id, desc))
            processed_item_ids.add(item_id)

        items_not_in_db_table = set(items_to_load) - processed_item_ids
        if items_not_in_db_table:
            for item_id in items_not_in_db_table:
                missing_items_info.append((item_id, None))

        if missing_items_info:
            if self.tokenizer is None or self.bert_model is None:
                self.tokenizer, self.bert_model = self.load_bert()
                if hasattr(self.bert_model, 'config') and hasattr(self.bert_model.config, 'dim'):
                    k_bert_dim = self.bert_model.config.dim

            for item_id, desc in tqdm(missing_items_info, desc="   Computing BERT/Zeros"):
                if item_id not in self.item_emb_dict:
                    bert_emb = self.get_embedding(desc)
                    if bert_emb.shape[0] != k_bert_dim:
                        bert_emb = np.zeros(k_bert_dim)
                    self.item_emb_dict[item_id] = bert_emb
        
        all_rows_users = []
        users_list_for_query = list(users_to_load)
        for i in range(0, len(users_list_for_query), CHUNK_SIZE):
            chunk = users_list_for_query[i : i + CHUNK_SIZE]
            if not chunk: continue
            try:
                cur.execute("""
                    SELECT "UserID", "UserEmbeddingVector"
                    FROM "User"
                    WHERE "UserID" = ANY(%s)
                """, (chunk,))
                rows_chunk = cur.fetchall()
                all_rows_users.extend(rows_chunk)
            except Exception as e:
                print(f"❌ Error querying user chunk starting at index {i}: {e}")
                raise

        self.user_emb_dict = {}
        processed_user_ids = set()
        k_dim_for_user = k_bert_dim
        for user_id, emb_vec in all_rows_users:
            if emb_vec is not None:
                try:
                    loaded_emb = np.array(emb_vec, dtype=float)
                    if len(loaded_emb) == k_dim_for_user:
                        self.user_emb_dict[user_id] = loaded_emb
                    else:
                        self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
                except Exception as e:
                    print(f"⚠️ User {user_id}: Error converting embedding: {e}. Using zeros.")
                    self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
            else:
                self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
            processed_user_ids.add(user_id)

        users_not_in_db_table = set(users_to_load) - processed_user_ids
        if users_not_in_db_table:
            print(f"   ⚠️ {len(users_not_in_db_table)} users from ratings were NOT found in 'User' table. Will use zero embeddings.")
            for user_id in users_not_in_db_table:
                self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)

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
                    print(f"Inferred K={self.k} from loaded user factors.")

        if self.k is None:
            raise ValueError("❌ Cannot determine K from loaded factors. Model load failed.")

        n_users_loaded = len(self.users)
        n_items_loaded = len(self.items_list)
        
        cur.execute('SELECT "AverageRating" FROM "Model" WHERE "ModelID"=%s', (model_id,))
        mu_row = cur.fetchone()
        self.mu = float(mu_row[0]) if mu_row and mu_row[0] is not None else 3

        self.model = self.UEIEModel(n_users_loaded, n_items_loaded, self.k, self.mu).to(self.device)
        self.model.eval()

        # with torch.no_grad():
        #     loaded_params = 0
        #     for user_id, bias in user_biases.items():
        #         if user_id in self.user2idx:
        #             self.model.b_u.weight[self.user2idx[user_id]] = bias
        #             loaded_params +=1
        #     for item_id, bias in item_biases.items():
        #         if item_id in self.item2idx:
        #             self.model.b_i.weight[self.item2idx[item_id]] = bias
        #             loaded_params +=1
        #     for user_id, factors in user_factors.items():
        #          if user_id in self.user2idx:
        #             if len(factors) == self.k:
        #                 self.model.P.weight[self.user2idx[user_id]] = torch.tensor(factors, dtype=torch.float32, device=self.device)
        #                 loaded_params += self.k
        #     for item_id, factors in item_factors.items():
        #         if item_id in self.item2idx:
        #             if len(factors) == self.k:
        #                 self.model.Q.weight[self.item2idx[item_id]] = torch.tensor(factors, dtype=torch.float32, device=self.device)
        #                 loaded_params += self.k
        print(f"✅ Loaded Factors/Biases into PyTorch Model for Prediction.")

    def load_bert(self):
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.bert_model.eval()
            self.bert_model.to(self.device)
            print("BERT model loaded.")
            return self.tokenizer, self.bert_model
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
            self.tokenizer, self.bert_model = None, None
            raise

    def get_embedding(self, text):
        if self.tokenizer is None or self.bert_model is None:
            print("      ⚠️ BERT model not loaded. Returning zero vector.")
            return np.zeros(768)

        if not text or not isinstance(text, str):
            k_bert_dim = self.bert_model.config.dim
            return np.zeros(k_bert_dim)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = self.bert_model(**inputs)
            except Exception as e:
                print(f"      ⚠️ Error during BERT forward pass for text '{text[:50]}...': {e}")
                k_bert_dim = self.bert_model.config.dim
                return np.zeros(k_bert_dim)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        embedding = mean_pooled.squeeze().cpu().numpy()
        return embedding

    def compute_inferred_preferences(self):
        print("   Computing inferred preferences (z_ui)...")
        if not self.ratings:
            print("⚠️ No ratings loaded. Cannot compute z_ui.")
            return np.array([])
        if not self.user_emb_dict or not self.item_emb_dict:
            print("⚠️ User or Item embedding dict is empty. Using zeros for z_ui.")
            return np.zeros(len(self.ratings))

        inferred = np.zeros(len(self.ratings))
        errors = 0
        k_dim_check = -1

        try:
            k_dim_check = len(next(iter(self.item_emb_dict.values())))
        except StopIteration:
            print("⚠️ Item embedding dict is empty after load! Using zeros for z_ui.")
            return np.zeros(len(self.ratings))

        for idx, (u, i, _) in enumerate(tqdm(self.ratings, desc="   Calculating z_ui")):
            user_vec = self.user_emb_dict.get(u)
            item_vec = self.item_emb_dict.get(i)

            valid = True
            if user_vec is None:
                errors += 1
                valid = False
            elif user_vec.shape[0] != k_dim_check:
                errors += 1
                valid = False

            if item_vec is None:
                errors += 1
                valid = False
            elif item_vec.shape[0] != k_dim_check:
                errors += 1
                valid = False

            if valid:
                try:
                    inferred[idx] = np.dot(user_vec, item_vec)
                except ValueError as e:
                    inferred[idx] = 0.0
                    errors += 1
            else:
                inferred[idx] = 0.0

        if errors > 0:
            print(f"⚠️ Encountered {errors} issues (missing/mismatched embeddings) during z_ui calculation.")

        if np.isnan(inferred).any() or np.isinf(inferred).any():
            print("⚠️ Inferred preferences contain NaN/Inf. Replacing with 0 before scaling.")
            inferred = np.nan_to_num(inferred, nan=0.0, posinf=0.0, neginf=0.0)

        if np.ptp(inferred) > 1e-9:
            scaler = MinMaxScaler(feature_range=(1, 5))
            try:
                scaled_inferred = scaler.fit_transform(inferred.reshape(-1, 1)).flatten()
                return scaled_inferred
            except ValueError as e:
                print(f"⚠️ Error scaling z_ui: {e}. Using clipped unscaled values.")
                return np.clip(inferred, 1.0, 5.0)
        else:
            mu_val = self.mu if self.mu != 0.0 else 3.0
            print(f"⚠️ Inferred preferences have no variance. Scaling skipped. Using constant value: {mu_val:.2f}")
            return np.full(len(self.ratings), mu_val)

    def train_model(self, epochs=500, batch_size=256):
        if not self.ratings: print("⚠️ No training data. Skipping."); return
        if self.model is None: print("⚠️ Model not initialized. Skipping."); return
        if self.optimizer is None: print("⚠️ Optimizer not initialized. Skipping."); return

        print(f"🚀 Starting training: {epochs} epochs, batch_size={batch_size}, device={self.device}")
        self.model.train()
        n = len(self.ratings)
        prev_avg_loss = float('inf')
        tolerance = 1e-6

        try:
            user_indices = [self.user2idx[u] for u, _, _ in self.ratings]
            item_indices = [self.item2idx[i] for _, i, _ in self.ratings]
            user_idx_all = torch.tensor(user_indices, device=self.device, dtype=torch.long)
            item_idx_all = torch.tensor(item_indices, device=self.device, dtype=torch.long)
            r_all = torch.tensor([r for _, _, r in self.ratings], dtype=torch.float32, device=self.device)
        except KeyError as e:
            print(f"❌ ERROR preparing tensors: User or Item ID '{e}' missing from index map! Aborting training.")
            return

        if self.inferred_prefs is None or len(self.inferred_prefs) != n:
            z_all = torch.full((n,), self.mu, dtype=torch.float32, device=self.device)
        else:
            z_all = torch.tensor(self.inferred_prefs, dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            total_main_loss_sum = 0.0

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
                current_batch_size = len(u_batch)

                r_hat = self.model(u_batch, i_batch)

                err_r = r_batch - r_hat
                err_z = z_batch - r_hat
                main_loss_batch = 0.5 * (err_r**2 + self.weight * err_z**2)
                main_loss_mean = main_loss_batch.mean()

                loss = main_loss_mean
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_main_loss_sum += main_loss_mean.item() * current_batch_size

            avg_main_loss = total_main_loss_sum / n

            with torch.no_grad():
                reg_p = torch.sum(self.model.P.weight**2)
                reg_q = torch.sum(self.model.Q.weight**2)
                reg_bu = torch.sum(self.model.b_u.weight**2)
                reg_bi = torch.sum(self.model.b_i.weight**2)
            
            final_reg_loss = 0.5 * self.lam * (reg_p + reg_q + reg_bu + reg_bi).item()
            avg_total_loss_to_track = avg_main_loss + final_reg_loss / n
            
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_total_loss_to_track:.4f}")

            if abs(prev_avg_loss - avg_total_loss_to_track) < tolerance:
                print(f"✅ Early stopping at epoch {epoch+1} (|Δloss|={abs(prev_avg_loss - avg_total_loss_to_track):.6e} < {tolerance})")
                break
            prev_avg_loss = avg_total_loss_to_track

        print("🏁 Training finished.")

    def write_model_to_db(self):
        if self.train_mode != 'train': print("Not in train mode. Skipping save."); return
        if self.connection is None: raise ValueError("❌ DB connection required.")
        if self.model is None: print("⚠️ Model not trained. Cannot save."); return
        
        cur = self.connection.cursor()
        self.model.eval()
        
        update_count_u, update_count_i = 0, 0
        try:
            for user_id in self.users:
                if user_id in self.user_emb_dict:
                    emb = self.user_emb_dict[user_id]
                    cur.execute('UPDATE "User" SET "UserEmbeddingVector"=%s, "modified_at"=NOW() WHERE "UserID"=%s', (emb.tolist(), user_id))
                    update_count_u += cur.rowcount
            for item_id in self.items_list:
                if item_id in self.item_emb_dict:
                    emb = self.item_emb_dict[item_id]
                    cur.execute('UPDATE "Item" SET "ItemEmbeddingVector"=%s, "modified_at"=NOW() WHERE "ItemID"=%s', (emb.tolist(), item_id))
                    update_count_i += cur.rowcount
            print(f"Updated embeddings for {update_count_u} users, {update_count_i} items.")
            self.connection.commit()
        except Exception as e:
            print(f"❌ Error updating embeddings: {e}")
            self.connection.rollback()
            return

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
                UPDATE "Model" SET "AverageRating"=%s, "modified_at"=NOW() WHERE "ModelID"=%s 
            """, (float(self.mu), self.model_id))
            
            self.connection.commit()
            print(len(self.user2idx))
            print(len(self.item2idx))


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
            model_id=1,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        
        # print(model.predict("cce9b2a37665aa5aa978d2dc622066ba", "7824768", 1))
        
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