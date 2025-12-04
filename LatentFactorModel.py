
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import psycopg2
from typing import List, Tuple, Union

class LatentFactorModel:
    def __init__(self, ratings, item_texts, connection=None):
        if ratings is None and item_texts is None and connection is None:
            raise ValueError("Either ratings and item_texts or connection must be provided.")
        if (ratings is not None or item_texts is not None) and connection is not None:
            raise ValueError("Provide either ratings and item_texts or connection, not both.")
        
        if connection is None:
            self.connection = None
            self.ratings = ratings
            self.item_texts = item_texts
            self.tokenizer, self.bert_model = self.load_bert()
            self.item_emb_dict = self.build_item_embeddings()
            self.k = next(iter(self.item_emb_dict.values())).shape[0]
            self.user_emb_dict, self.users = self.build_user_embeddings(self.ratings, self.item_emb_dict)
            self.n_users = len(self.users)
            self.n_items = len(self.item_texts)
            self.P, self.Q, self.b_u, self.b_i, self.mu = self.init_latent_model()
            self.inferred_prefs = self.compute_inferred_preferences()
        else:
            self.connection = connection
            self.load_model_from_db()

    def load_bert(self):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        model.eval()
        return tokenizer, model

    def get_embedding(self, text):
        if not text:
            return np.zeros(768)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]
        return token_embeddings.mean(dim=0).numpy()

    def build_item_embeddings(self):    
        item_emb_dict = {item: self.get_embedding(text) \
                        for item, text in self.item_texts.items()}
        return item_emb_dict

    def build_user_embeddings(self, ratings, item_emb_dict):
        user_ratings = defaultdict(list)
        for user, item, rating in ratings:
            user_ratings[user].append((item, rating))

        user_emb_dict = {}
        for user, item_ratings in user_ratings.items():
            # personalized liking thresholding
            avg_rating = np.mean([r for _, r in item_ratings])
            liked_items = [item for item, r in item_ratings if r > avg_rating]
            if not liked_items:
                user_emb_dict[user] = np.zeros(self.k)
            else:
                vecs = [item_emb_dict[item] for item in liked_items]
                # average
                agg = np.mean(vecs, axis=0)
                user_emb_dict[user] = agg
        return user_emb_dict, list(user_ratings.keys())

    def init_latent_model(self):
        # latent vectors of users
        P = {user: 0.01 * np.random.randn(self.k) for user in self.users}
        # latent vectors of items
        Q = {item: 0.01 * np.random.randn(self.k) for item in self.item_texts.keys()}
        # bias of users
        b_u = {user: 0.0 for user in self.users}
        # bias of items
        b_i = {item: 0.0 for item in self.item_texts.keys()}
        # average rating μ
        mu = np.mean([r for _, _, r in self.ratings])
        return P, Q, b_u, b_i, mu

    def compute_prediction(self, user, item):
        # r^_ui = μ + bias_user + bias_item + latent_user · latent_item
        return self.mu + self.b_u[user] + self.b_i[item] + np.dot(self.P[user], self.Q[item])

    def gradients(self, user, item, r_ui, z_ui, weight, lam):
        r_hat = self.compute_prediction(user, item)
        # e_r = r_ui - r^_ui
        e_r = r_ui - r_hat
        # e_t = z_ui - r^_ui
        e_t = z_ui - r_hat
        # Q_ui = e_r + w * e_t
        Q_ui = e_r + weight * e_t

        # grad latent user  = -(Q_ui * latent_item) + λ * latent_user
        grad_pu = -(Q_ui * self.Q[item]) + lam * self.P[user]
        # grad latent item  = -(Q_ui * latent_user) + λ * latent_item
        grad_qi = -(Q_ui * self.P[user]) + lam * self.Q[item]
        # grad bias user    = -Q_ui + λ * bias_user
        grad_bu = -Q_ui + lam * self.b_u[user]
        # grad bias item    = -Q_ui + λ * bias_item
        grad_bi = -Q_ui + lam * self.b_i[item]

        return grad_pu, grad_qi, grad_bu, grad_bi

    def compute_inferred_preferences(self):
        inferred_prefs = np.zeros(len(self.ratings))
        for idx, (user, item, _) in enumerate(self.ratings):
            inferred_prefs[idx] = np.dot(self.user_emb_dict[user], self.item_emb_dict[item])
        scaler = MinMaxScaler(feature_range=(1, 5))
        inferred_prefs = scaler.fit_transform(inferred_prefs.reshape(-1, 1)).flatten().astype(float)
        return inferred_prefs

    def compute_objective(self, weight, lam):
        total_loss = 0.0
        for idx, (user, item, r_ui) in enumerate(self.ratings):
            r_hat = self.compute_prediction(user, item)
            z_ui = self.inferred_prefs[idx]

            # e_r = r_ui - r^_ui
            e_r = r_ui - r_hat
            # e_t = z_ui - r^_ui
            e_t = z_ui - r_hat

            # ((r_ui - r^_ui)^2 + w * (z_ui - r^_ui)^2) / 2
            total_loss += (e_r**2 + weight * (e_t**2)) / 2.0

        # Tikhonov regularization
        reg = (lam / 2.0) * (
            sum(np.sum(p**2) for p in self.P.values()) +
            sum(np.sum(q**2) for q in self.Q.values()) +
            sum(b**2 for b in self.b_u.values()) +
            sum(b**2 for b in self.b_i.values())
        )

        return total_loss + reg
    
    def train(self, max_epochs=500, lr=0.001, weight=0.3, lam=0.01, tol=1e-6, save=False, connection=None):
        if save and conn is None:
            raise ValueError("Database connection is required to save the model.")

        prev_obj = float("inf")
        for _ in range(max_epochs):
            indices = np.arange(len(self.ratings))
            np.random.shuffle(indices)

            for idx in indices:
                user, item, r_ui = self.ratings[idx]
                z_ui = self.inferred_prefs[idx]

                grad_pu, grad_qi, grad_bu, grad_bi = self.gradients(
                    user, item, r_ui, z_ui, weight, lam
                )

                # SGD update
                self.P[user] -= lr * grad_pu
                self.Q[item] -= lr * grad_qi
                self.b_u[user] -= lr * grad_bu
                self.b_i[item] -= lr * grad_bi

            obj = self.compute_objective(weight, lam)
            if abs(prev_obj - obj) < tol:
                break
            
            prev_obj = obj
        
        if save and conn is not None:
            self.write_model_to_db(conn, lr, weight, lam)

    def predict_rating(self, user, item):
        return self.compute_prediction(user, item)
    
    def write_model_to_db(self, lr=0.001, weight=0.3, lam=0.01):
        cur = self.connection.cursor()

        cur.execute(
            'INSERT INTO "Model" ("NumFactors", "GlobalBias", "LearningRate", "RegularizationRate", "InferredPrefWeight") '
            'VALUES (%s, %s, %s, %s, %s) RETURNING "ModelID"',
            (self.k, float(self.mu), lr, lam, weight)
        )
        model_id = cur.fetchone()[0]

        user_ratings = defaultdict(list)
        for user, item, rating in self.ratings:
            user_ratings[user].append((item, rating))
        
        user_avg_dict = {}
        for user, item_ratings in user_ratings.items():
            avg_rating = np.mean([r for _, r in item_ratings])
            user_avg_dict[user] = avg_rating
        
        for user in self.users:
            user_emb = self.user_emb_dict[user].tolist()
            cur.execute(
                'UPDATE "User" SET "UserEmbeddingVector" = %s, "LikingThreshold" = %s WHERE "UserID" = %s',
                (user_emb, float(user_avg_dict[user]), user)
            )
            
            cur.execute(
                'INSERT INTO "UserFactor" ("UserBias", "UserFactors", "ModelID", "UserID") '
                'VALUES (%s, %s, %s, %s) ',
                (float(self.b_u[user]), self.P[user].tolist(), model_id, user)
            )

        for item in self.item_texts.keys():
            item_emb = self.item_emb_dict[item].tolist()
            cur.execute(
                'UPDATE "Item" SET "ItemEmbeddingVector" = %s WHERE "ItemID" = %s',
                (item_emb, item)
            )
            
            cur.execute(
                'INSERT INTO "ItemFactor" ("ItemBias", "ItemFactors", "ModelID", "ItemID") '
                'VALUES (%s, %s, %s, %s) ',
                (float(self.b_i[item]), self.Q[item].tolist(), model_id, item)
            )
            
        self.connection.commit()
        
    def load_model_from_db(self):
        cur = self.connection.cursor()
    
        self.tokenizer, self.bert_model = self.load_bert()
        cur.execute('SELECT * FROM "Item"')
        rows = cur.fetchall()
        self.item_texts = {row[0]: row[2] for row in rows}
        self.item_emb_dict = {}
        # load item embeddings from db if exist, otherwise compute
        for row in rows:
            item_id = row[0]
            description = row[2]
            emb_from_db = row[3]
            if emb_from_db is not None:
                embedding = np.array(emb_from_db, dtype=float)
            else:
                embedding = self.get_embedding(description)
            self.item_emb_dict[item_id] = embedding
        
        self.k = next(iter(self.item_emb_dict.values())).shape[0]
        
        cur.execute('SELECT * FROM "get_all_unique_ratings"()')
        self.ratings = []
        rows = cur.fetchall()
        for row in rows:
            self.ratings.append((row[1], row[2], float(row[0])))
            
        # take the user embeddings from db if exist, otherwise compute
        self.user_emb_dict = {}
        cur.execute('SELECT "UserID", "UserEmbeddingVector" FROM "User"')
        for row in cur.fetchall():
            user_id = row[0]
            emb_from_db = row[1]
            if emb_from_db is not None:
                embedding = np.array(emb_from_db, dtype=float)
            else:
                user_ratings = [(u, i, r) for u, i, r in self.ratings if u == user_id]
                if user_ratings:
                    user_emb_dict, _ = self.build_user_embeddings(user_ratings, self.item_emb_dict)
                    embedding = user_emb_dict[user_id]
                else:
                    embedding = np.zeros(self.k)
            self.user_emb_dict[user_id] = embedding

        # rebuild user embedding for users that have new ratings
        new_user_ids = set()
        cur.execute('SELECT * from "get_new_ratings"()')
        rows = cur.fetchall()
        for row in rows:
            # userid, itemid, rating
            new_user_ids.add(row[1])

        for user_id in new_user_ids:
            cur.execute("""
                SELECT DISTINCT ON (r."UserID", r."ItemID")
                    r."UserID",
                    r."ItemID",
                    r."RatingValue"
                FROM "Rating" r
                WHERE r."UserID" = %s
                ORDER BY r."UserID", r."ItemID", r."created_at" DESC;
            """, (user_id,))
            rows = cur.fetchall()
            
            ratings_for_user = [(r[0], r[1], float(r[2])) for r in rows]  # (user, item, rating)
            user_emb_dict, _ = self.build_user_embeddings(ratings_for_user, self.item_emb_dict)
            embedding = user_emb_dict[user_id]
            
            avg_rating = np.mean([r[2] for r in ratings_for_user])
            
            self.user_emb_dict[user_id] = embedding
            
            cur.execute(
                'UPDATE "User" SET "UserEmbeddingVector" = %s, "LikingThreshold" = %s WHERE "UserID" = %s',
                (embedding.tolist(), float(avg_rating), user_id)
            )
            
        self.users = list(self.user_emb_dict.keys())
        self.n_users = len(self.users)
        self.n_items = len(self.item_texts)
        self.P, self.Q, self.b_u, self.b_i, self.mu = self.init_latent_model()
        self.inferred_prefs = self.compute_inferred_preferences()
        self.connection.commit()
    
if __name__ == "__main__":
    np.random.seed(42)
    
    # ratings = [
    #     ("u0", "i0", 4), ("u0", "i1", 3), ("u0", "i2", 2), ("u0", "i3", 3), ("u0", "i4", 5), ("u0", "i5", 1), ("u0", "i6", 4),
    #     ("u1", "i0", 2), ("u1", "i1", 2), ("u1", "i2", 3), ("u1", "i3", 4), ("u1", "i5", 1), ("u1", "i7", 5), ("u1", "i8", 3),
    #     ("u2", "i0", 1), ("u2", "i1", 4), ("u2", "i2", 5), ("u2", "i3", 5), ("u2", "i4", 2), ("u2", "i6", 3), ("u2", "i9", 4),
    #     ("u3", "i1", 5), ("u3", "i2", 4), ("u3", "i5", 3), ("u3", "i7", 2), ("u3", "i8", 4), ("u3", "i9", 5),
    #     ("u4", "i0", 3), ("u4", "i3", 2), ("u4", "i4", 4), ("u4", "i5", 5), ("u4", "i6", 1), ("u4", "i8", 3),
    #     ("u5", "i1", 4), ("u5", "i2", 2), ("u5", "i4", 3), ("u5", "i7", 5), ("u5", "i9", 1),
    #     ("u6", "i0", 5), ("u6", "i2", 3), ("u6", "i3", 4), ("u6", "i5", 2), ("u6", "i8", 4),
    #     ("u7", "i1", 3), ("u7", "i3", 5), ("u7", "i4", 2), ("u7", "i6", 4), ("u7", "i7", 3), ("u7", "i9", 5),
    # ]

    # item_texts = {
    #     "i0": "item_0 is a bestselling science fiction book, full of drama and creativity",
    #     "i1": "item_1 is a famous action movie with spectacular scenes",
    #     "i2": "item_2 is a new pop music album by a globally renowned artist",
    #     "i3": "item_3 is a high-tech product, a smartphone with advanced features",
    #     "i4": "item_4 is an exciting adventure video game with stunning graphics",
    #     "i5": "item_5 is a delicious dish from Asian cuisine, rich in flavor",
    #     "i6": "item_6 is a nature documentary film, inspiring environmental protection",
    #     "i7": "item_7 is a self-help book that improves personal skills and positive thinking",
    #     "i8": "item_8 is a high-end fashion product, stylish and modern",
    #     "i9": "item_9 is an online programming course, suitable for beginners"
    # }
    
    DB_NAME = "rsystem"
    USER = "flyingcat2003"
    HOST = "localhost"
    PASSWORD = "Hanly1912a"
    
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=USER,
            host=HOST,
            password=PASSWORD
        )
    except:
        print("I am unable to connect to the database")
    
    ratings = []
    item_texts = {}
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM \"Rating\"")
        rows = cur.fetchall()
        for row in rows:
            ratings.append((row[3], row[4], float(row[1])))
            
        cur.execute("SELECT * FROM \"Item\"")
        rows = cur.fetchall()
        for row in rows:
            item_texts[row[0]] = row[2]
    
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    
    model = LatentFactorModel(train_ratings, item_texts)
    model.train(max_epochs=500, lr=0.001, weight=0.3, lam=0.01, save=False)
    # for (u, i, r) in test_ratings:
    #     pred = model.predict_rating(u, i)
    #     print(f"{u}-{i}: true={r}, pred={pred:.2f}")
        
    def compute_rmse(model, ratings_set):
        squared_error = 0.0
        count = 0
        for user, item, r_ui in ratings_set:
            pred = model.predict_rating(user, item)
            squared_error += (r_ui - pred) ** 2
            if count < 50:
                print(f"{user}-{item}: true={r_ui}, pred={pred:.2f}")
                count += 1
        mse = squared_error / len(ratings_set)
        rmse = np.sqrt(mse)
        return rmse
    
    def compute_mae(model, ratings_set):
        absolute_error = 0.0
        for user, item, r_ui in ratings_set:
            pred = model.predict_rating(user, item)
            absolute_error += abs(r_ui - pred)
        mae = absolute_error / len(ratings_set)
        return mae
    
    test_rmse = compute_rmse(model, test_ratings)
    print(f"RMSE on test set: {test_rmse:.4f}")

    test_mae = compute_mae(model, test_ratings)
    print(f"MAE on test set: {test_mae:.4f}")
    
    # item_factor = {}
    # user_factor = {}
    # bias_u = {}
    # bias_i = {}
    # global_bias = 0.0
    
    # with conn.cursor() as cur:
    #     cur.execute('SELECT "UserID", "UserBias", "UserFactors" FROM "UserFactor" WHERE "UserBias" IS NOT NULL AND "ModelID" = 17')
    #     rows = cur.fetchall()
    #     for row in rows:
    #         bias_u[row[0]] = row[1]
    #         user_factor[row[0]] = row[2]
        
    #     cur.execute('SELECT "ItemID", "ItemBias", "ItemFactors" FROM "ItemFactor" WHERE "ItemBias" IS NOT NULL AND "ModelID" = 17')
    #     rows = cur.fetchall()
    #     for row in rows:
    #         bias_i[row[0]] = row[1]
    #         item_factor[row[0]] = row[2]
            
    #     cur.execute('SELECT "GlobalBias" FROM "Model" ORDER BY "ModelID" DESC LIMIT 1')
    #     row = cur.fetchone()
    #     global_bias = row[0]
    
    # def predict(user, item):
    #     if user not in user_factor or item not in item_factor:
    #         return None
    #     p_u = np.array(user_factor[user])
    #     q_i = np.array(item_factor[item])
    #     b_u = bias_u[user]
    #     b_i = bias_i[item]
    #     return float(global_bias) + float(b_u) + float(b_i) + np.dot(p_u, q_i)
    
    # print(predict(1259, 1544))
    # count = 0
    # for item_id, text in item_texts.items():
    #     print(item_id, ":", text)
    #     count += 1
    #     if count == 10:
    #         break
    # print(item_texts)

    def save_predictions_to_db(
        cur,
        predictions: List[Tuple[int, int, float]],
        table_name: str = "Predict"
    ) -> int:
        if not predictions:
            print("⚠️ Không có dữ liệu dự đoán để lưu.")
            return 0
        
        # SQL UPSERT statement: Chèn, nếu đã tồn tại (dựa trên PK/Unique Index), thì cập nhật Value.
        sql_upsert = f"""
            INSERT INTO goodreads."{table_name}" ("UserID", "ItemID", "Value")
            VALUES (%s, %s, %s)
            ON CONFLICT ("UserID", "ItemID") DO UPDATE SET
                "Value" = EXCLUDED."Value";
        """
        try:
            # Sử dụng execute_batch để chèn hàng loạt, nhanh hơn nhiều so với vòng lặp execute
            psycopg2.extras.execute_batch(cur, sql_upsert, predictions)
            
            row_count = cur.rowcount
            cur.commit()
            
            print(f"✅ Đã lưu/cập nhật thành công {row_count} dự đoán vào bảng '{table_name}'.")
            return row_count
            
        except Exception as e:
            print(f"❌ Lỗi khi lưu dự đoán vào DB: {e}")
            cur.rollback()  
            raise
    
    conn.close()