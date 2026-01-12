import psycopg2
import psycopg2.extras
from tqdm import tqdm
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .ReviewRating import LatentFactorModel as ReviewModel
    from .UCInit import LatentFactorModel as UCInitModel
    from .UEIE import LatentFactorModel as UEIEModel
    from .IInit import LatentFactorModel as IInitModel
except ImportError:
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
        domain_id=None,       
        model_ids_map=None,
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
        self.domain_id = domain_id
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
        self.model_id = model_ids_map.get("PLA")
        self.train_mode = train_mode.lower()
        self.model_ids_map = model_ids_map
        
        if self.connection is None:
            raise ValueError("❌ DB connection required.")
        if self.domain_id is None:
            raise ValueError("❌ 'domain_id' is required.")
        if model_ids_map is None:
            raise ValueError("❌ 'model_ids_map' is required.")

        self.review_model = ReviewModel(
            connection=self.connection,
            train_mode='load',
            model_id=model_ids_map["ReviewRating"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ueie_model = UEIEModel(
            connection=self.connection,
            train_mode='load',
            model_id=model_ids_map["UEIE"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01,
            weight=0.3
        )
        self.ucinit_model = UCInitModel(
            connection=self.connection,
            train_mode='load',
            model_id=model_ids_map["UCInit"],
            domain_id=self.domain_id,
            k=90,
            lr=0.001,
            lam=0.01
        )

        self.iinit_models = [None] * 9
        for i in range(1, 9):
            self.iinit_models[i] = IInitModel(
                connection=self.connection,
                train_mode='load',
                model_id=model_ids_map[f"IInit {i}"],
                domain_id=self.domain_id,
                k=90,
                lr=0.001,
                lam=0.01,
                interaction_type_id=i
            )

        self.load_user_item_from_db()

        # freeze submodels (important!)
        for param in self.review_model.parameters():
            param.requires_grad = False
        for param in self.ueie_model.parameters():
            param.requires_grad = False
        for param in self.ucinit_model.parameters():
            param.requires_grad = False
        for i in range(1, 9):
            for param in self.iinit_models[i].parameters():
                param.requires_grad = False

        # -----------------------------
        # PLA-specific learnable parameters
        # -----------------------------
        self.num_models = 11  # number of submodels
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

    def load_user_item_from_db(self):
        cur = self.connection.cursor()
        try: 
            cur.execute("""
            SELECT "Id"
            FROM "User"
            WHERE "DomainId" = %s
            ORDER BY "Id"
            """, (self.domain_id,))
            rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error querying users: {e}")
            raise

        self.users = [str(row[0]) for row in rows]
        
        try:
            cur.execute("""
                SELECT "Id"
                FROM "Item"
                WHERE "DomainId" = %s
                ORDER BY "Id"
                """, (self.domain_id,))
            rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error querying items: {e}")
            raise
        self.items = [str(row[0]) for row in rows]

    def load_theta_from_db(self):
        """
        Load tham số từ DB. Nếu gặp NULL hoặc sai kích thước -> Tự động Init Random.
        """
        cursor = self.connection.cursor()
        
        model_order_names = ["ReviewRating", "UEIE", "UCInit", "IInit 1", "IInit 2", "IInit 3", "IInit 4", "IInit 5", "IInit 6", "IInit 7", "IInit 8"]
        thetas = []
        expected_dim = 2 * self.k

        print("--- Loading Theta from DB ---")
        for name in model_order_names:
            mid = self.model_ids_map[name]
            cursor.execute('SELECT "LearnableParameters" FROM "Model" WHERE "Id" = %s', (mid,))
            result = cursor.fetchone()
            
            arr = np.random.randn(expected_dim).astype(np.float32) * 0.1
            status = "🆕 NULL/Empty -> Random Init"

            if result and result[0] is not None:
                try:
                    db_arr = np.array(result[0], dtype=np.float32)
                    
                    if db_arr.size == expected_dim:
                        arr = db_arr
                        status = "✅ Loaded"
                    else:
                        status = f"⚠️ Size Mismatch ({db_arr.size} vs {expected_dim}) -> Re-init"
                except Exception:
                    status = "❌ Parse Error -> Re-init"
            
            print(f"   - {name} (ID {mid}): {status}")
            thetas.append(arr)

        theta_matrix = np.stack(thetas, axis=0)
        
        self.theta = nn.Parameter(torch.tensor(theta_matrix, dtype=torch.float32, device=self.device))
        
        if self.theta.shape != torch.Size([self.num_models, expected_dim]):
            raise RuntimeError(f"❌ Theta Shape Error: Got {self.theta.shape}, Expected [{self.num_models}, {expected_dim}]")

    def load_ratings_from_db(self, limit_total=1000000, ratio=0.8):
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
            raise

        if not rows:
            raise ValueError("❌ No rating data loaded from DB.")

        total = len(rows)
        if total == 0:
            raise ValueError("❌ Zero ratings loaded from DB.")

        # Filter ratings to only include users/items that exist in all submodels
        all_ratings = [(str(u), str(i), float(r)) for u, i, r in rows]
        filtered_ratings = []
        
        for u, i, r in all_ratings:
            # Check if user and item exist in all submodel mappings
            if (u in self.review_model.user2idx and i in self.review_model.item2idx and
                u in self.ueie_model.user2idx and i in self.ueie_model.item2idx and
                u in self.ucinit_model.user2idx and i in self.ucinit_model.item2idx and
                u in self.iinit_models[1].user2idx and i in self.iinit_models[1].item2idx and
                u in self.iinit_models[2].user2idx and i in self.iinit_models[2].item2idx and
                u in self.iinit_models[3].user2idx and i in self.iinit_models[3].item2idx and
                u in self.iinit_models[4].user2idx and i in self.iinit_models[4].item2idx and
                u in self.iinit_models[5].user2idx and i in self.iinit_models[5].item2idx and
                u in self.iinit_models[6].user2idx and i in self.iinit_models[6].item2idx and
                u in self.iinit_models[7].user2idx and i in self.iinit_models[7].item2idx and
                u in self.iinit_models[8].user2idx and i in self.iinit_models[8].item2idx):
                filtered_ratings.append((u, i, r))
        
        if not filtered_ratings:
            raise ValueError("❌ No valid ratings found after filtering for existing users/items in submodels.")
        
        self.ratings = filtered_ratings
        self.test_ratings = []
        print(f"✅ Loaded {len(self.ratings)}/{total} ratings (filtered for valid users/items in submodels)")

    def forward(self, u, i):
        # ---- 1. Get phi(u, i) ----
        Pu = self.review_model.model.P(torch.tensor([self.review_model.user2idx[u]], dtype=torch.long, device=self.device)).squeeze(0)     # shape [k]
        Qi = self.review_model.model.Q(torch.tensor([self.review_model.item2idx[i]], dtype=torch.long, device=self.device)).squeeze(0)     # shape [k]
        phi = torch.cat([Pu, Qi], dim=-1)   # shape [2k]

        # ---- 2. Get predictions from each submodel ----
        preds = []
        
        # Các model đơn lẻ
        preds.append(torch.tensor(self.review_model.predict(u, i, 1), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(self.ueie_model.predict(u, i, 0), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(self.ucinit_model.predict(u, i, 0), dtype=torch.float32, device=self.device))
        
        # Các model IInit (Loop qua list)
        for model in self.iinit_models:
            if model is not None:
                p_val = model.predict(u, i, 0)
                preds.append(torch.tensor(p_val, dtype=torch.float32, device=self.device))
            # else:
            #     # Fallback nếu model bị None (đề phòng)
            #     preds.append(torch.tensor(self.ucinit_model.mu, dtype=torch.float32, device=self.device))

        r_s = torch.stack(preds)   # Shape sẽ là [11] khớp với theta

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
    
    def fit(self, n_epochs=500, batch_size=256, tol=1e-6):
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

        for item in self.items:
            for user in self.users:
                self.predict(user, item, 0)
              
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
            self.write_predictions_to_db(user, item, r_pred.item())

            # if input is a single (user, item), return scalar
            if r_pred.numel() == 1:
                return r_pred.item()
            else:
                return r_pred
            
    def write_predictions_to_db(self, user, item, value, table_name="Predict"):
        if self.connection is None:
            raise ValueError("❌ DB connection required.")

        cur = self.connection.cursor()
        try:
            cur.execute(f"""
                INSERT INTO "{table_name}" ("UserId", "ItemId", "Value")
                VALUES (%s, %s, %s)
                ON CONFLICT ("UserId", "ItemId")
                DO UPDATE SET
                "Value" = EXCLUDED."Value";
            """, (user, item, value))
            self.connection.commit()
            print(f"✅ Prediction written.")
        except Exception as e:
            print(f"❌ Error writing prediction to DB: {e}")
            self.connection.rollback()
        finally:
            cur.close()

    def write_model_to_db(self):
        if self.connection is None: return
        cur = self.connection.cursor()
        self.eval()

        theta_values = self.theta.detach().cpu().numpy() # Shape [self.num_models, 2*self.k]
        model_order_names = ["ReviewRating", "UEIE", "UCInit", "IInit 1", "IInit 2", "IInit 3", "IInit 4", "IInit 5", "IInit 6", "IInit 7", "IInit 8"]
        data_to_insert = []
        
        for idx, name in enumerate(model_order_names):
            mid = self.model_ids_map[name]
            theta_list = theta_values[idx].tolist()
            data_to_insert.append((theta_list, mid))

        try:
            query = 'UPDATE "Model" SET "LearnableParameters" = %s, "ModifiedAt" = NOW() WHERE "Id" = %s;'
            psycopg2.extras.execute_batch(cur, query, data_to_insert)
            self.connection.commit()
            print(f"💾 Saved {len(data_to_insert)} theta vectors to DB (replaced NULLs).")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            self.connection.rollback()
        finally:
            cur.close()


    # def load_theta_from_db(self):
    #     cursor = self.connection.cursor()

    #     # Define correct order: ReviewRating(3), UEIE(1), UCInit(2), IInit(4)
    #     model_order = [self.model_ids_map["ReviewRating"], self.model_ids_map["UEIE"], self.model_ids_map["UCInit"], self.model_ids_map["IInit"]]
    #     thetas = []
    #     for mid in model_order:
    #         cursor.execute("""
    #             SELECT "LearnableParameters" FROM "Model" WHERE "Id" = %s;
    #         """,
    #             (mid,)
    #         )
    #         result = cursor.fetchone()
    #         if result is None:
    #             raise ValueError(f"❌ No LearnableParameters found for ModelID={mid}")

    #         # PostgreSQL REAL[] → Python list → NumPy array → torch tensor
    #         arr = np.array(result[0], dtype=np.float32)
    #         thetas.append(arr)

    #     # Stack to shape [num_models, 2*k]
    #     theta_matrix = np.stack(thetas, axis=0)
    #     theta_tensor = torch.tensor(theta_matrix, dtype=torch.float32, device=self.device)

    #     # Assign as nn.Parameter
    #     self.theta = nn.Parameter(theta_tensor)

    #     print(f"✅ Loaded theta from DB, shape: {self.theta.shape}")


    # def write_model_to_db(self):
    #     if self.train_mode != 'train': print("Not in train mode. Skipping save."); return
    #     if self.connection is None: raise ValueError("❌ DB connection required.")

    #     cur = self.connection.cursor()
    #     self.eval()

    #     # Ensure parameters are on CPU and converted to numpy
    #     theta_values = self.theta.detach().cpu().numpy()  # shape [num_models, 2*k]
    #     # model_names = ["UEIE", "UCInit", "ReviewRating", "IInit"]
    #     # model_names = ["ReviewRating", "UEIE", "UCInit", "IInit"]
    #     model_ids = [3, 1, 2, 4]

    #     # Prepare data tuples for batch insert
    #     data_to_insert = []
    #     for _, (id, theta_row) in enumerate(zip(model_ids, theta_values)):
    #         theta_list = theta_row.tolist()  # convert to Python list (for REAL[])
    #         data_to_insert.append((theta_list, id))

    #     # Execute batch insert/update
    #     cur = self.connection.cursor()
    #     query = """
    #         UPDATE "Model" SET "LearnableParameters" = %s
    #         WHERE "Id" = %s;
    #     """

    #     psycopg2.extras.execute_batch(cur, query, data_to_insert)
    #     self.connection.commit()
    #     cur.close()

    #     print(f"[INFO] Saved {len(data_to_insert)} theta vectors to 'Model' table.")
