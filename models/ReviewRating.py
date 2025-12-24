import time
import numpy as np
from google import genai
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
        print(f"Using device: {self.device}")
        
        self.connection = connection
        if self.connection is None:
            raise ValueError("❌ DB connection is required for all modes.")

        self.weight = weight
        self.lam = lam
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.k = k
        
        self.ratings_reviews = None
        self.test_ratings = None
        self.users = []
        self.items_list = []
        self.user2idx = {}
        self.item2idx = {}
        self.mu = 0.0
        self.model = None
        self.optimizer = None
        self.client = genai.Client(api_key="AIzaSyAma-rodMYxbC_jBQWxtwPrFof8tyEivws")
        self.domain_id = domain_id

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

        else: # train_mode == 'train'
            if self.k is None:
                raise ValueError("❌ Parameter 'k' (e.g., 90) is required for training.")
            
            self.load_user_item_from_db()
            self.user2idx = {u: idx for idx, u in enumerate(self.users)}
            self.item2idx = {i: idx for idx, i in enumerate(self.items_list)}
            
            if ratings_and_reviews is None:
                self.load_ratings_from_db()
            else:
                all_data = [(u,i,r,c) for u,i,r,c in ratings_and_reviews if u in self.user2idx and i in self.item2idx]
                if not all_data:
                    raise ValueError("❌ No valid ratings (all filtered out).")
                # self.ratings_reviews, self.test_ratings_reviews = train_test_split(all_data, test_size=0, random_state=42)
                self.ratings_reviews, self.test_ratings_reviews = all_data, []
                

            if not self.ratings_reviews:
                raise ValueError("❌ No training ratings available (hoặc tất cả rating đã bị lọc bỏ).")

            self.mu = np.mean([r for _, _, r, _ in self.ratings_reviews]) if self.ratings_reviews else 3.5
            
            n_users_final = len(self.users)
            n_items_final = len(self.items_list)
            
            if n_users_final == 0 or n_items_final == 0:
                raise ValueError(f"❌ Cannot initialize model with {n_users_final} users / {n_items_final} items.")
            
            self.model = self.ReviewRatingModel(n_users_final, n_items_final, self.k, self.mu).to(self.device)

            self.optimizer = optim.AdamW(self.model.parameters(), 
                                         lr=self.lr, 
                                         weight_decay=self.lam)
            print("✅ Initialization complete and ready for training.")

    class ReviewRatingModel(nn.Module):
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
        except Exception as e:
            raise
            
        self.users = [str(row[0]) for row in rows]
        
        try:
            cur.execute("""
                SELECT "Id"
                From "Item"
                WHERE "DomainId" = %s
                ORDER BY "Id"
            """, (self.domain_id,))
            rows = cur.fetchall()
        except Exception as e:
            raise
        
        self.items_list = [str(row[0]) for row in rows]
        
    def load_or_convert_reviews(self, limit_total=1000000):
        cur = self.connection.cursor()
        try:
            cur.execute("""
                SELECT "Id", "UserId", "ItemId", "Value", "ReviewText", "ConvertedScore"
                FROM "Rating"
                WHERE "DomainId" = %s
                LIMIT %s
            """, (self.domain_id, limit_total))
            rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error loading reviews: {e}")
            return []

        results = []
        for rid, uid, iid, rating, text, converted in tqdm(rows, desc="Processing Reviews"):
            if converted is None:
                # converted = self.review_to_rating(text)
                converted = 3.0
                try:
                    cur.execute("""
                        UPDATE "Rating" 
                        SET "ConvertedScore" = %s
                        WHERE "Id" = %s
                    """, (converted, rid))
                    self.connection.commit()
                except Exception as e:
                    print(f"⚠️ Error updating ConvertedScore for RatingID={rid}: {e}")
                    self.connection.rollback()
            
            results.append((uid, iid, rating, converted))

        self.ratings_reviews_dict = {(u, i): (r, c) for u, i, r, c in results}
        
    def review_to_rating(self, review_text):
        if str(review_text).strip() == "" or review_text == None:
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
        except Exception as e:
            print(f"⚠️ Error while rating review: {e}")
            rating = 0

        time.sleep(12)
        
        return rating

    def load_ratings_from_db(self, limit_total=1000000, ratio=0.8):
        cur = self.connection.cursor()
        try:
            cur.execute("""
                SELECT "Id", "UserId", "ItemId", "Value", "ReviewText", "ConvertedScore"
                FROM "Rating"
                WHERE "DomainId" = %s
                LIMIT %s
            """, (self.domain_id, limit_total))
            rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error loading reviews: {e}")
            raise
        
        if not rows:
            raise ValueError("❌ No rating data loaded from DB.")

        results = []
        for rid, uid, iid, rating, text, converted in tqdm(rows, desc="Processing Reviews"):
            uid = str(uid)
            iid = str(iid)
            if uid not in self.user2idx or iid not in self.item2idx:
                continue

            if converted is None:
                # converted = self.review_to_rating(text)
                converted = 3.0
                try:
                    cur.execute("""
                        UPDATE "Rating" 
                        SET "ConvertedScore" = %s
                        WHERE "Id" = %s
                    """, (converted, rid))
                    self.connection.commit()
                except Exception as e:
                    print(f"⚠️ Error updating ConvertedScore for RatingID={rid}: {e}")
                    self.connection.rollback()
            
            results.append((uid, iid, rating, converted))

        total = len(results)
        if total == 0:
             raise ValueError("❌ Zero ratings loaded (or all were filtered out).")
        
        train_count = int(total * ratio)
        if train_count == 0: train_count = total
             
        self.ratings_reviews = [(str(u), str(i), float(r), float(c)) for u, i, r, c in results[:train_count]]
        self.test_ratings_reviews = [(str(u), str(i), float(r), float(c)) for u, i, r, c in results[train_count:]]

    def train_model(self, epochs=500, batch_size=512):
        if not self.ratings_reviews: print("⚠️ No training data. Skipping."); return
        if self.model is None: print("⚠️ Model not initialized. Skipping."); return
        if self.optimizer is None: print("⚠️ Optimizer not initialized. Skipping."); return

        print(f"🚀 Starting training: {epochs} epochs, batch_size={batch_size}, device={self.device}")
        self.model.train()
        n = len(self.ratings_reviews)
        prev_avg_loss = float('inf')
        tolerance = 1e-6

        try:
            user_indices = [self.user2idx[u] for u, _, _, _ in self.ratings_reviews]
            item_indices = [self.item2idx[i] for _, i, _, _ in self.ratings_reviews]
            user_idx_all = torch.tensor(user_indices, device=self.device, dtype=torch.long)
            item_idx_all = torch.tensor(item_indices, device=self.device, dtype=torch.long)
            r_all = torch.tensor([r for _, _, r, _ in self.ratings_reviews], dtype=torch.float32, device=self.device)
        except KeyError as e:
            print(f"❌ ERROR preparing tensors: User or Item ID '{e}' missing from index map! Aborting training.")
            return

        z_all = torch.tensor([c for _, _, _, c in self.ratings_reviews], dtype=torch.float32, device=self.device)
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
            
            # print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_total_loss_to_track:.4f}")

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

        try:
            with torch.no_grad():
                all_user_biases = self.model.b_u.weight.squeeze().cpu().numpy()
                all_user_factors = self.model.P.weight.cpu().numpy()
                all_item_biases = self.model.b_i.weight.squeeze().cpu().numpy()
                all_item_factors = self.model.Q.weight.cpu().numpy()

            user_factor_data = []
            item_factor_data = []
            
            users_in_train = set(u for u, _, _, _ in self.ratings_reviews)
            items_in_train = set(i for _, i, _, _ in self.ratings_reviews)

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

        except Exception as e:
            print(f"   ❌ Error saving factors/biases (UPSERT stage): {e}")
            self.connection.rollback()

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
        print(f"   Found {len(rows_items)} items in 'Item' table.")

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
                    print(f"   Inferred K={self.k} from loaded user factors.")

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
                
        cur.execute('SELECT "AverageRating" FROM "Model" WHERE "Id"=%s', (model_id,))
        mu_row = cur.fetchone()
        self.mu = float(mu_row[0]) if mu_row and mu_row[0] is not None else 3
        
        self.model = self.ReviewRatingModel(
            n_users_loaded, n_items_loaded, self.k, self.mu,
            P_init_arr, Q_init_arr, b_u_init_arr, b_i_init_arr
        ).to(self.device)
        self.model.eval()

        print(f"✅ Loaded Factors/Biases into PyTorch Model for Prediction.")
    
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