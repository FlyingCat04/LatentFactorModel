import time
import numpy as np
from google import genai
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
import torch.optim as optim

class LatentFactorModel(nn.Module):
    """
    Latent Factor Model integrating Explicit Ratings and Review Texts (ReviewRating).
    
    This model enhances traditional matrix factorization by incorporating sentiment 
    scores derived from textual reviews (processed via LLM - Google Gemini) alongside 
    numerical ratings.
    """
    def __init__(
        self,
        db_config=None,
        ratings_and_reviews=None, 
        items=None,
        k=90,
        weight=0.3,
        lam=0.01,
        lr=0.001,
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
        
        # Initialize Gemini AI client for review sentiment analysis
        self.client = genai.Client(api_key="AIzaSyAma-rodMYxbC_jBQWxtwPrFof8tyEivws")
        
        # System variables
        self.ratings_reviews = []
        self.test_ratings_reviews = []
        self.users = []
        self.items_list = []
        self.user2idx = {}
        self.item2idx = {}
        self.mu = 3.5 
        
        self.model = None
        self.optimizer = None

        if self.db_config is None:
            raise ValueError("Database configuration is required.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("train_mode must be strictly 'train' or 'load'.")

        self._initialize_system(ratings_and_reviews)

    def _initialize_system(self, input_ratings_reviews):
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
            
            if input_ratings_reviews is None:
                self.load_ratings_from_db()
            else:
                valid_data = [(u, i, r, c) for u, i, r, c in input_ratings_reviews if u in self.user2idx and i in self.item2idx]
                self.ratings_reviews = valid_data if valid_data else []
                self.test_ratings_reviews = []

            # Handle cold-start scenario where no ratings are available
            if not self.ratings_reviews:
                self.mu = 3.5 
            else:
                self.mu = np.mean([r for _, _, r, _ in self.ratings_reviews])
            
            n_users_final = len(self.users)
            n_items_final = len(self.items_list)
            
            # Initialize Neural Network Architecture
            self.model = self.ReviewRatingModel(n_users_final, n_items_final, self.k, self.mu).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.lam)

    class ReviewRatingModel(nn.Module):
        """Neural embedding layer framework for Latent Factor optimization."""
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
            
            # Inject pre-computed factors or fallback to Xavier initialization securely
            with torch.no_grad():
                if P_init is not None:
                    t = torch.tensor(P_init, dtype=torch.float32)
                    if t.shape == self.P.weight.shape:
                        self.P.weight.copy_(t)
                    else:
                        nn.init.xavier_uniform_(self.P.weight)
                else:
                    nn.init.xavier_uniform_(self.P.weight)

                if Q_init is not None:
                    t = torch.tensor(Q_init, dtype=torch.float32)
                    if t.shape == self.Q.weight.shape:
                        self.Q.weight.copy_(t)
                    else:
                        nn.init.xavier_uniform_(self.Q.weight)
                else:
                    nn.init.xavier_uniform_(self.Q.weight)

                if b_u_init is not None:
                    t = torch.tensor(b_u_init, dtype=torch.float32).unsqueeze(1)
                    if t.shape == self.b_u.weight.shape:
                        self.b_u.weight.copy_(t)
                    else:
                        nn.init.zeros_(self.b_u.weight)
                else:
                    nn.init.zeros_(self.b_u.weight)

                if b_i_init is not None:
                    t = torch.tensor(b_i_init, dtype=torch.float32).unsqueeze(1)
                    if t.shape == self.b_i.weight.shape:
                        self.b_i.weight.copy_(t)
                    else:
                        nn.init.zeros_(self.b_i.weight)
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
        
    def review_to_rating(self, review_text):
        """
        Invokes Google Gemini LLM to perform sentiment analysis on a textual review 
        and map it to a numerical score (1-5).
        """
        if not review_text or str(review_text).strip() == "":
            return 0
        
        prompt = f"""
                Given the title and text body of a review, analyze its sentiment and return a rating from 1 to 5, where
                0 = The review is irrelevant to the product, meaningless, spam, or contains nonsense text.
                1 = Very negative,
                2 = Negative,
                3 = Neutral or mixed,
                4 = Positive,
                5 = Very positive.
                - If the review meaning can be understood or clearly intended (even with typos), rate it normally (1–5).  
                - Assign 0 only when the text has no clear meaning, no relation to the product, or is obvious spam.
                
                Only return the number.

                Input:
                - Review: "{review_text}"
                """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            rating = int(response.text.strip())
        except Exception:
            rating = 0

        # Delay to comply with free-tier API rate limits
        time.sleep(12)
        return rating

    def load_ratings_from_db(self, limit_total=1000000, ratio=0.8):
        """Retrieves explicit ratings and evaluates sentiment for textual reviews if uncalculated."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT "Id", "UserId", "ItemId", "Value", "ReviewText", "ConvertedScore"
                        FROM "Rating"
                        WHERE "DomainId" = %s AND "Value" IS NOT NULL
                        LIMIT %s
                    """, (self.domain_id, limit_total))
                    rows = cur.fetchall()
                    
                    if not rows:
                        self.ratings_reviews = []
                        self.test_ratings_reviews = []
                        return

                    results = []
                    for rid, uid, iid, rating, text, converted in rows:
                        uid = str(uid)
                        iid = str(iid)
                        if uid not in self.user2idx or iid not in self.item2idx:
                            continue

                        # Generate ConvertedScore via LLM if it's missing in DB
                        if converted is None:
                            converted = self.review_to_rating(text)
                            try:
                                cur.execute('UPDATE "Rating" SET "ConvertedScore" = %s WHERE "Id" = %s', (converted, rid))
                                conn.commit()
                            except Exception:
                                conn.rollback()
                        
                        results.append((uid, iid, rating, converted))
        except Exception:
            self.ratings_reviews = []
            self.test_ratings_reviews = []
            return

        total = len(results)
        if total == 0:
             self.ratings_reviews = []
             self.test_ratings_reviews = []
             return
        
        train_count = int(total * ratio)
        if train_count == 0: train_count = total
             
        self.ratings_reviews = [(str(u), str(i), float(r), float(c)) for u, i, r, c in results[:train_count]]
        self.test_ratings_reviews = [(str(u), str(i), float(r), float(c)) for u, i, r, c in results[train_count:]]

    def load_model_from_db(self, model_id):
        """Loads structural weights and parameters for inference."""
        item_factors, item_biases = {}, {}
        user_factors, user_biases = {}, {}
        temp_k_from_factors = None
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
            raise RuntimeError(f"Failed to load model state from database: {e}")

        if temp_k_from_factors is not None:
            self.k = temp_k_from_factors

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
                
        self.model = self.ReviewRatingModel(
            n_users_loaded, n_items_loaded, self.k, self.mu,
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

        user_factor_data = []
        item_factor_data = []
        
        users_in_train = set(u for u, _, _, _ in self.ratings_reviews) if self.ratings_reviews else set()
        items_in_train = set(i for _, i, _, _ in self.ratings_reviews) if self.ratings_reviews else set()

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
            pass # Suppressed to maintain background task stability

    # -------------------------------------------------------------------------
    # Training and Inference
    # -------------------------------------------------------------------------
    def train_model(self, epochs=500, batch_size=512):
        """Execute the optimization loop using Mini-Batch Gradient Descent."""
        if not self.ratings_reviews or self.model is None: 
            return

        self.model.train()
        n = len(self.ratings_reviews)
        tolerance = 1e-6
        prev_avg_loss = float('inf')

        # Convert interaction data to PyTorch Tensors
        try:
            user_indices = [self.user2idx[u] for u, _, _, _ in self.ratings_reviews]
            item_indices = [self.item2idx[i] for _, i, _, _ in self.ratings_reviews]
            user_idx_all = torch.tensor(user_indices, device=self.device, dtype=torch.long)
            item_idx_all = torch.tensor(item_indices, device=self.device, dtype=torch.long)
            
            # r_all: Explicit ratings, z_all: Textual sentiment scores
            r_all = torch.tensor([r for _, _, r, _ in self.ratings_reviews], dtype=torch.float32, device=self.device)
            z_all = torch.tensor([c for _, _, _, c in self.ratings_reviews], dtype=torch.float32, device=self.device)
        except KeyError:
            return

        for epoch in range(epochs):
            total_main_loss_sum = 0.0
            
            # Data Shuffling
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
                
                # Forward Pass
                r_hat = self.model(u_batch, i_batch)

                # Composite loss computation (Rating Error + Review Sentiment Error)
                err_r = r_batch - r_hat
                err_z = z_batch - r_hat
                main_loss_batch = 0.5 * (err_r**2 + self.weight * err_z**2)
                main_loss_mean = main_loss_batch.mean()

                # Backward Pass
                self.optimizer.zero_grad()
                main_loss_mean.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_main_loss_sum += main_loss_mean.item() * len(u_batch)

            # L2 Regularization & Early Stopping evaluation
            avg_main_loss = total_main_loss_sum / n
            with torch.no_grad():
                reg_p = torch.sum(self.model.P.weight**2)
                reg_q = torch.sum(self.model.Q.weight**2)
                reg_bu = torch.sum(self.model.b_u.weight**2)
                reg_bi = torch.sum(self.model.b_i.weight**2)
            
            final_reg_loss = 0.5 * self.lam * (reg_p + reg_q + reg_bu + reg_bi).item()
            avg_total_loss_to_track = avg_main_loss + final_reg_loss / n
            
            if abs(prev_avg_loss - avg_total_loss_to_track) < tolerance:
                break
            prev_avg_loss = avg_total_loss_to_track

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