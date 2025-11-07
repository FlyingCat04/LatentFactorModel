import psycopg2
import psycopg2.extras
from tqdm import tqdm
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ReviewRating import LatentFactorModel as ReviewModel
from UCInit import LatentFactorModel as UCInitModel
from UEIE import LatentFactorModel as UEIEModel
from IInit import LatentFactorModel as IInitModel

class PLA(nn.Module):
    def __init__(
        self,
        # ratings=None,
        # theta=None,
        # users = None,
        # items=None,
        connection=None, 
        k=90,
        lambda_reg=0.01,
        lambda_pri=0.01,
        lr=0.002,
        model_id=-1,
        train_mode='train',
        device=None,
):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.connection = connection
        self.ratings = None
        self.test_ratings = None
        self.theta = None
        # self.users = users
        # self.items = items
        # self.n_users = len(self.users)
        # self.n_items = len(self.items)
        # self.k_dim_bert = 768
        self.k = k
        self.lambda_reg = lambda_reg
        self.lambda_pri = lambda_pri
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()

        self.review_model = ReviewModel(
            connection=conn,
            train_mode='load',
            model_id=3,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ueie_model = UEIEModel(
            connection=conn,
            train_mode='load',
            model_id=1,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ucinit_model = UCInitModel(
            connection=conn,
            train_mode='load',
            model_id=2,
            k=90,
            lr=0.001,
            lam=0.01
        )
        self.iinit_model = IInitModel(
            connection=conn,
            train_mode='load',
            model_id=4,
            k=90,
            lr=0.001,
            lam=0.01
        )
        # self.ueie_model = self.review_model
        # self.ucinit_model = self.review_model
        # self.iinit_model = self.review_model

        # freeze submodels (important!)
        for param in self.review_model.parameters():
            param.requires_grad = False
        for param in self.ueie_model.parameters():
            param.requires_grad = False
        for param in self.ucinit_model.parameters():
            param.requires_grad = False
        for param in self.iinit_model.parameters():
            param.requires_grad = False

        # -----------------------------
        # PLA-specific learnable parameters
        # -----------------------------
        self.num_models = 4  # number of submodels
        self.theta = nn.Parameter(torch.randn(self.num_models, 2 * k) * 0.1)  # learnable combination weights
        self.bias = nn.Parameter(torch.zeros(1))  # optional bias

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lambda_reg)
        self.loss_fn = nn.MSELoss()

        if self.connection is None:
            raise ValueError("❌ DB connection required.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("❌ train_mode must be 'train' or 'load'")
        if self.k is None:
            raise ValueError("❌ Parameter 'k' (e.g., 90) is required for training.")            
        
        self.load_ratings_from_db()
        if not self.ratings:
            raise ValueError("❌ No training ratings available.")

        self.load_theta_from_db()
        if self.theta is None or self.theta.numel() == 0:
            raise ValueError("❌ No learnable parameters available.")

        # if self.train_mode == 'load':
        #     pass
        # else:
        #     pass

    def load_theta_from_db(self):
        cursor = self.connection.cursor()

        # Define correct order: ReviewRating(3), UEIE(1), UCInit(2), IInit(4)
        model_order = [3, 1, 2, 4]
        thetas = []
        for mid in model_order:
            cursor.execute("""
                SELECT "LearnableParameter" FROM "Model" WHERE "ModelID" = %s;
            """,
                (mid,)
            )
            result = cursor.fetchone()
            if result is None:
                raise ValueError(f"❌ No LearnableParameter found for ModelID={mid}")

            # PostgreSQL REAL[] → Python list → NumPy array → torch tensor
            arr = np.array(result[0], dtype=np.float32)
            thetas.append(arr)

        # Stack to shape [num_models, 2*k]
        theta_matrix = np.stack(thetas, axis=0)
        theta_tensor = torch.tensor(theta_matrix, dtype=torch.float32, device=self.device)

        # Assign as nn.Parameter
        self.theta = nn.Parameter(theta_tensor)

        print(f"✅ Loaded theta from DB, shape: {self.theta.shape}")

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
        if train_count == 0: train_count = total

        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]

    def forward(self, u, i):
        # ---- 1. Get phi(u, i) ----
        Pu = self.review_model.model.P(torch.tensor([self.review_model.user2idx[u]], dtype=torch.long, device=self.device)).squeeze(0)     # shape [k]
        Qi = self.review_model.model.Q(torch.tensor([self.review_model.item2idx[i]], dtype=torch.long, device=self.device)).squeeze(0)     # shape [k]
        phi = torch.cat([Pu, Qi], dim=-1)   # shape [2k]

        # ---- 2. Get predictions from each submodel ----
        r_review = torch.tensor(self.review_model.predict(u, i, 1), dtype=torch.float32, device=self.device)
        r_ueie   = torch.tensor(self.ueie_model.predict(u, i, 0), dtype=torch.float32, device=self.device)
        r_ucinit = torch.tensor(self.ucinit_model.predict(u, i, 0), dtype=torch.float32, device=self.device)
        r_iinit  = torch.tensor(self.iinit_model.predict(u, i, 0), dtype=torch.float32, device=self.device)
        r_s = torch.stack([r_review, r_ueie, r_ucinit, r_iinit])   # shape [S]

        # ---- 3. Compute weight logits and softmax ----
        logits = torch.mv(self.theta, phi)   # [S]
        alphas = F.softmax(logits, dim=0)    # [S]

        # ---- 4. Weighted combination ----
        r_hat = torch.sum(alphas * r_s) + self.bias.to(self.device)  # final rating prediction

        # ---- 5. Return outputs ----
        return r_hat, alphas, r_s

    def compute_loss(self, r_true, r_hat, alphas):
        # Eq. (15)
        main_loss = 0.5 * (r_true - r_hat) ** 2
        diff = alphas[1:] - alphas[0]
        pri_loss = 0.5 * torch.sum(torch.clamp(diff, min=0.0) ** 2)
        reg_loss = 0.5 * torch.sum(self.theta ** 2)
        total = main_loss + self.lambda_pri * pri_loss + self.lambda_reg * reg_loss
        return total, main_loss, pri_loss, reg_loss

    def train_step(self, batch):
        """
        Perform one training step on a batch of (u, i, r) triples.

        Args:
            batch: list of tuples (u, i, r_true)
            optimizer: torch optimizer
        Returns:
            dict of losses for logging
        """
        self.train()
        self.optimizer.zero_grad()
        total_loss, total_main, total_pri, total_reg = 0.0, 0.0, 0.0, 0.0

        for (u, i, r_true) in batch:
            # Ensure tensors on correct device
            # u = torch.tensor([u], dtype=torch.long, device=self.device)
            # i = torch.tensor([i], dtype=torch.long, device=self.device)
            # u = str(u)
            # i = str(i)
            r_true = torch.tensor(r_true, dtype=torch.float32, device=self.device)

            r_hat, alphas, r_s = self.forward(u, i)  # Forward
            loss, main_loss, pri_loss, reg_loss = self.compute_loss(r_true, r_hat, alphas)  # Compute loss

            total_loss += loss
            total_main += main_loss
            total_pri += pri_loss
            total_reg += reg_loss

        # ---- 6. Average and Backprop ----
        n = len(batch)
        total_loss = total_loss / n
        total_main = total_main / n
        total_pri = total_pri / n
        total_reg = total_reg / n

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "main": total_main.item(),
            "priority": total_pri.item(),
            "reg": total_reg.item()
        }
    
    def fit(self, n_epochs=10, batch_size=256, tol=1e-6):
        """
        Train the PLA model using mini-batch gradient descent with early stopping
        if the change in total loss between epochs is less than `tol`.
        """
        self.train()
        n_samples = len(self.ratings)
        n_batches = (n_samples + batch_size - 1) // batch_size

        prev_loss = None
        print("🚀 Start training PLA model...")
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            epoch_main = 0.0
            epoch_pri = 0.0
            epoch_reg = 0.0

            # Optional: shuffle data for robustness
            # random.shuffle(self.ratings)

            for b in range(n_batches):
                batch = self.ratings[b * batch_size:(b + 1) * batch_size]
                if not batch:
                    continue

                losses = self.train_step(batch)
                epoch_loss += losses["loss"]
                epoch_main += losses["main"]
                epoch_pri += losses["priority"]
                epoch_reg += losses["reg"]

            # ---- Average per epoch ----
            epoch_loss /= n_batches
            epoch_main /= n_batches
            epoch_pri /= n_batches
            epoch_reg /= n_batches

            print(
                f"Epoch {epoch:02d}/{n_epochs} | "
                f"Loss: {epoch_loss:.6f} | "
                f"Main: {epoch_main:.6f} | "
                f"Priority: {epoch_pri:.6f} | "
                f"Reg: {epoch_reg:.6f}"
            )

            # ---- Early stopping condition ----
            if prev_loss is not None:
                delta = abs(prev_loss - epoch_loss)
                if delta < tol:
                    print(f"🛑 Early stopping at epoch {epoch} (ΔLoss={delta:.2e} < {tol})")
                    break

            prev_loss = epoch_loss

        print("✅ Training finished! θ parameters are now learned.")

    def predict(self, user, item, p):
        self.eval()  # set model to evaluation mode (important!)
        with torch.no_grad():  # disable gradient tracking

            # # ensure user/item are tensors on correct device
            # if not torch.is_tensor(user):
            #     user = torch.tensor([user], dtype=torch.long, device=self.device)
            # else:
            #     user = user.to(self.device)
            # if not torch.is_tensor(item):
            #     item = torch.tensor([item], dtype=torch.long, device=self.device)
            # else:
            #     item = item.to(self.device)
            # if p == 1:
            #     print(self.review_model.user2idx.get(user), self.review_model.item2idx.get(item))

            # forward pass
            r_pred, _, _ = self.forward(user, item)  # [batch] shape

            # if input is a single (user, item), return scalar
            if r_pred.numel() == 1:
                return r_pred.item()
            else:
                return r_pred

    def write_model_to_db(self):
        if self.train_mode != 'train': print("Not in train mode. Skipping save."); return
        if self.connection is None: raise ValueError("❌ DB connection required.")

        cur = self.connection.cursor()
        self.eval()

        # Ensure parameters are on CPU and converted to numpy
        theta_values = self.theta.detach().cpu().numpy()  # shape [num_models, 2*k]
        # model_names = ["UEIE", "UCInit", "ReviewRating", "IInit"]
        # model_names = ["ReviewRating", "UEIE", "UCInit", "IInit"]
        model_ids = [3, 1, 2, 4]

        # Prepare data tuples for batch insert
        data_to_insert = []
        for _, (id, theta_row) in enumerate(zip(model_ids, theta_values)):
            theta_list = theta_row.tolist()  # convert to Python list (for REAL[])
            data_to_insert.append((theta_list, id))

        # Execute batch insert/update
        cur = self.connection.cursor()
        query = """
            UPDATE "Model" SET "LearnableParameter" = %s
            WHERE "ModelID" = %s;
        """

        psycopg2.extras.execute_batch(cur, query, data_to_insert)
        self.connection.commit()
        cur.close()

        print(f"[INFO] Saved {len(data_to_insert)} theta vectors to 'Model' table.")

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
        print("✅ Database connected successfully.")
    except Exception as e:
        print(f"❌ DB Connection Error: {e}")
        exit()

    try:
        model = PLA(
            connection=conn,
            train_mode='train',
        )
        
        # print(model.predict("95b22f99934e50fa545d40099e3986e2", "12907847", 1))

        test_ratings_to_use = model.test_ratings if model.test_ratings is not None else []
        if not test_ratings_to_use:
             print("⚠️ No test ratings loaded. Evaluation will be skipped.")

        if model is not None:
            print("\n--- Starting Model Training ---")
            model.fit()
        else:
            print("❌ Model initialization failed. Skipping training.")

        if model is not None:
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
                        pred = model_instance.predict(str(user), str(item), 1)
                    else:
                        pred = model_instance.predict(str(user), str(item), 0)
                    squared_error += (r_ui - pred) ** 2
                    if count < 50:
                        print(f"   {user}-{item}: true={r_ui}, pred={pred:.2f}")
                        count += 1
                if not ratings_set: return 0.0
                mse = squared_error / len(ratings_set)
                rmse = np.sqrt(mse)
                return rmse

            # def compute_mae(model_instance, ratings_set):
            #     if not ratings_set: return float('nan')
            #     absolute_error = 0.0
            #     for user, item, r_ui in tqdm(ratings_set, desc="MAE Eval"):
            #         pred = model_instance.predict(user, item, 1)
            #         absolute_error += abs(float(r_ui) - float(pred))
            #     if not ratings_set: return 0.0
            #     mae = absolute_error / len(ratings_set)
            #     return mae

            test_rmse = compute_rmse(model, test_ratings_to_use)
            print(f"RMSE on test set: {test_rmse:.4f}")

            # test_mae = compute_mae(model, test_ratings_to_use)
            # print(f"MAE on test set: {test_mae:.4f}")
        
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












        # x, y = "95b22f99934e50fa545d40099e3986e2", "12907847"
        # print(model.review_model.predict(x, y, 1))
        # print(1)

        # u = torch.tensor([model.review_model.user2idx[x]], dtype=torch.long)
        # i = torch.tensor([model.review_model.item2idx[y]], dtype=torch.long)
        # print(2)

        # Pu = model.review_model.model.P(u).squeeze(0)   # shape [k]
        # Qi = model.review_model.model.Q(i).squeeze(0)   # shape [k]
        # print(3)

        # phi = torch.cat([Pu, Qi], dim=-1)   # shape [2k]
        # print(4)

        # print(len(Pu), Pu)
        # print(Qi)
        # print(phi)



    # def fit(self, train_data, n_epochs=10, batch_size=128):
    #     for epoch in range(n_epochs):
    #         random.shuffle(train_data)
    #         batches = [
    #             train_data[i:i + batch_size]
    #             for i in range(0, len(train_data), batch_size)
    #         ]
    #         epoch_loss = 0.0
    #         for batch in batches:
    #             logs = self.train_step(batch)
    #             epoch_loss += logs["loss"]
    #         print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss / len(batches):.4f}")



# class PLA(nn.Module):
#     def __init__(self, dim, m, lambda_pri=0.5, lambda_reg=0.01):
#         super().__init__()
#         self.m = m
#         self.theta = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(m)])
#         self.lambda_pri = lambda_pri
#         self.lambda_reg = lambda_reg

#     def forward(self, phi, sub_preds):
#         # Compute unnormalized scores
#         alpha_tilde = torch.stack([phi @ theta for theta in self.theta])  # shape (m,)
#         # Softmax normalization
#         alpha = F.softmax(alpha_tilde, dim=0)
#         # Aggregate prediction
#         r_hat = torch.sum(alpha * sub_preds)
#         return r_hat, alpha

#     def loss(self, r_true, r_hat, alpha):
#         L_main = 0.5 * (r_true - r_hat)**2
#         # Priority constraint loss
#         L_pri = 0.5 * torch.sum(F.relu(alpha[1:] - alpha[0])**2)
#         # Regularization loss
#         L_reg = 0.5 * sum(torch.sum(theta**2) for theta in self.theta)
#         # Total loss
#         return L_main + self.lambda_pri * L_pri + self.lambda_reg * L_reg

# model = PLA(dim=6, m=3)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# # Example data
# phi = torch.tensor([0.6, -0.2, 0.9, 0.4, 0.7, -0.1])
# sub_preds = torch.tensor([4.0, 3.8, 4.2])
# r_true = torch.tensor(4.5)

# # Forward
# r_hat, alpha = model(phi, sub_preds)
# loss = model.loss(r_true, r_hat, alpha)

# # Backward and update
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print("Pred:", r_hat.item(), "Loss:", loss.item())
# for i, t in enumerate(model.theta):
#     print(f"Theta_{i+1}:", t.data)
