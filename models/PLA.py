import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import psycopg2
import psycopg2.extras

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
    """
    Priority-based Learning Algorithm (PLA) Ensemble Model.
    
    This model dynamically combines predictions from multiple latent factor sub-models
    (ReviewRating, UEIE, UCInit, and 8 IInit interaction models) using an attention-like 
    mechanism (theta). It learns optimal weights based on user-item features (phi).
    """
    def __init__(
        self,
        db_config=None,
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

        if db_config is None:
            raise ValueError("Database configuration is required.")
        if domain_id is None:
            raise ValueError("Domain ID is required.")
        if model_ids_map is None:
            raise ValueError("Model IDs map is required.")

        self.db_config = db_config
        self.domain_id = domain_id
        self.model_ids_map = model_ids_map
        self.k = k
        self.lambda_reg = lambda_reg
        self.lambda_pri = lambda_pri
        self.lr = lr
        self.model_id = model_ids_map.get("PLA")
        self.train_mode = train_mode.lower()
        
        self.ratings = []
        self.test_ratings = None
        self.theta = None
        self.num_models = 11  # Base (3) + Interaction types (8)
        self.bias = nn.Parameter(torch.zeros(1))

        self._initialize_system()

    def _initialize_system(self):
        """Initializes sub-models, loads system data, and configures the optimizer."""
        self._load_submodels()
        self.load_user_item_from_db()
        
        # Freeze all parameters in sub-models as PLA only learns the combination weights
        for param in self.review_model.parameters(): param.requires_grad = False
        for param in self.ueie_model.parameters(): param.requires_grad = False
        for param in self.ucinit_model.parameters(): param.requires_grad = False
        for i in range(1, 9):
            if self.iinit_models[i]:
                for param in self.iinit_models[i].parameters():
                    param.requires_grad = False

        self.load_theta_from_db()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lambda_reg)
        self.loss_fn = nn.MSELoss()

        if self.train_mode == 'train':
            self.load_ratings_from_db()

    def _load_submodels(self):
        """Instantiates all 11 sub-models configured for inference."""
        self.review_model = ReviewModel(
            db_config=self.db_config,
            train_mode='load',
            model_id=self.model_ids_map["ReviewRating"],
            domain_id=self.domain_id,
            k=self.k, lr=0.001, lam=0.01, weight=0.3
        )
        self.ueie_model = UEIEModel(
            db_config=self.db_config,
            train_mode='load',
            model_id=self.model_ids_map["UEIE"],
            domain_id=self.domain_id,
            k=self.k, lr=0.001, lam=0.01, weight=0.3
        )
        self.ucinit_model = UCInitModel(
            db_config=self.db_config,
            train_mode='load',
            model_id=self.model_ids_map["UCInit"],
            domain_id=self.domain_id,
            k=self.k, lr=0.001, lam=0.01
        )

        self.iinit_models = [None] * 9
        for i in range(1, 9):
            try:
                self.iinit_models[i] = IInitModel(
                    db_config=self.db_config,
                    train_mode='load',
                    model_id=self.model_ids_map[f"IInit {i}"],
                    domain_id=self.domain_id,
                    k=self.k, lr=0.001, lam=0.01,
                    interaction_type_id=i
                )
            except Exception:
                self.iinit_models[i] = None

    # -------------------------------------------------------------------------
    # Database Operations (Just-in-Time Connection)
    # -------------------------------------------------------------------------
    def load_user_item_from_db(self):
        """Retrieves the complete list of Users and Items for the domain."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT "Id" FROM "User" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.users = [str(row[0]) for row in cur.fetchall()]
                    
                    cur.execute('SELECT "Id" FROM "Item" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.items = [str(row[0]) for row in cur.fetchall()]
        except Exception:
            self.users = []
            self.items = []

    def load_theta_from_db(self):
        """Loads the learned weighting parameters (theta) for each sub-model."""
        model_order_names = [
            "ReviewRating", "UEIE", "UCInit", 
            "IInit 1", "IInit 2", "IInit 3", "IInit 4", 
            "IInit 5", "IInit 6", "IInit 7", "IInit 8"
        ]
        thetas = []
        expected_dim = 2 * self.k

        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    for name in model_order_names:
                        mid = self.model_ids_map.get(name)
                        result = None
                        if mid:
                            cursor.execute('SELECT "LearnableParameters" FROM "Model" WHERE "Id" = %s', (mid,))
                            result = cursor.fetchone()
                        
                        # Initialize with small random noise
                        arr = np.random.randn(expected_dim).astype(np.float32) * 0.1

                        if result and result[0] is not None:
                            try:
                                db_arr = np.array(result[0], dtype=np.float32)
                                if db_arr.size == expected_dim:
                                    arr = db_arr
                            except Exception:
                                pass
                        
                        thetas.append(arr)
        except Exception:
            # Fallback to random thetas if DB connection fails
            thetas = [np.random.randn(expected_dim).astype(np.float32) * 0.1 for _ in model_order_names]

        theta_matrix = np.stack(thetas, axis=0)
        self.theta = nn.Parameter(torch.tensor(theta_matrix, dtype=torch.float32, device=self.device))

    def load_ratings_from_db(self, limit_total=1000000):
        """Loads interaction ratings used as ground truth to train the PLA weights."""
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
            return

        all_ratings = [(str(u), str(i), float(r)) for u, i, r in rows]
        
        valid_users = set(self.users)
        valid_items = set(self.items)
        
        # Filter out ratings with non-existent users or items
        self.ratings = [(u, i, r) for u, i, r in all_ratings if u in valid_users and i in valid_items]

    # -------------------------------------------------------------------------
    # Core Logic: Forward Pass & Loss
    # -------------------------------------------------------------------------
    def forward(self, u, i):
        """
        Computes the final predicted rating using an attention-based linear combination 
        of the 11 sub-models' predictions.
        """
        # 1. Extract context features phi(u, i) from the Review Model (Base)
        u_idx = self.review_model.user2idx.get(u)
        i_idx = self.review_model.item2idx.get(i)
        
        if u_idx is not None and u_idx < self.review_model.model.P.num_embeddings:
            Pu = self.review_model.model.P(torch.tensor([u_idx], device=self.device)).squeeze(0)
        else:
            Pu = torch.zeros(self.k, device=self.device) 

        if i_idx is not None and i_idx < self.review_model.model.Q.num_embeddings:
            Qi = self.review_model.model.Q(torch.tensor([i_idx], device=self.device)).squeeze(0)
        else:
            Qi = torch.zeros(self.k, device=self.device) 
            
        phi = torch.cat([Pu, Qi], dim=-1)

        # 2. Gather predictions from all sub-models
        preds = []
        
        def safe_predict(model):
            """Safely invokes a sub-model's prediction, falling back to mean rating if unavailable."""
            try:
                if model is None: 
                    return self.ucinit_model.mu
                val = model.predict(u, i, 0)
                if isinstance(val, (np.ndarray, list, torch.Tensor)):
                    val = float(val)
                return val
            except Exception:
                return self.ucinit_model.mu 

        # Base Sub-models
        preds.append(torch.tensor(safe_predict(self.review_model), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(safe_predict(self.ueie_model), dtype=torch.float32, device=self.device))
        preds.append(torch.tensor(safe_predict(self.ucinit_model), dtype=torch.float32, device=self.device))
        
        # Interaction Sub-models
        for idx in range(1, 9):
            val = safe_predict(self.iinit_models[idx])
            preds.append(torch.tensor(val, dtype=torch.float32, device=self.device))

        r_s = torch.stack(preds)

        # 3. Compute dynamic weights (alphas) via softmax
        logits = torch.mv(self.theta, phi)
        alphas = F.softmax(logits, dim=0)

        # 4. Compute weighted ensemble prediction
        r_hat = torch.sum(alphas * r_s) + self.bias.to(self.device)

        return r_hat, alphas, r_s

    def compute_loss(self, r_true, r_hat, alphas):
        """
        Calculates composite loss including MSE, priority smoothness, and L2 regularization.
        """
        main_loss = 0.5 * (r_true - r_hat) ** 2
        
        # Smoothness penalty for adjacent alpha weights
        diff = alphas[1:] - alphas[:-1]
        pri_loss = 0.5 * torch.sum(diff ** 2)
        
        reg_loss = 0.5 * torch.sum(self.theta ** 2)
        
        total_loss = main_loss + self.lambda_pri * pri_loss + self.lambda_reg * reg_loss
        return total_loss, main_loss, pri_loss, reg_loss

    # -------------------------------------------------------------------------
    # Training & Persistence
    # -------------------------------------------------------------------------
    def train_step(self, batch):
        """Executes a single optimization step for a mini-batch."""
        self.train()
        self.optimizer.zero_grad()
        total_loss, total_main, total_pri, total_reg = 0.0, 0.0, 0.0, 0.0

        for (u, i, r_true) in batch:
            r_true_tensor = torch.tensor(r_true, dtype=torch.float32, device=self.device)
            r_hat, alphas, r_s = self.forward(u, i)
            loss, main_loss, pri_loss, reg_loss = self.compute_loss(r_true_tensor, r_hat, alphas)

            total_loss += loss
            total_main += main_loss
            total_pri += pri_loss
            total_reg += reg_loss

        n = len(batch)
        if n == 0: 
            return {"loss": 0, "main": 0, "priority": 0, "reg": 0}

        avg_loss = total_loss / n
        avg_loss.backward()
        self.optimizer.step()

        return {
            "loss": avg_loss.item(),
            "main": (total_main / n).item(),
            "priority": (total_pri / n).item(),
            "reg": (total_reg / n).item()
        }
    
    def fit(self, n_epochs=500, batch_size=256, tol=1e-6):
        """Optimizes the ensemble weights (theta) over multiple epochs."""
        if not self.ratings:
            self.save_predictions_to_db()
            return

        self.train()
        n_samples = len(self.ratings)
        n_batches = (n_samples + batch_size - 1) // batch_size
        prev_loss = None

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            random.shuffle(self.ratings)

            for b in range(n_batches):
                batch = self.ratings[b * batch_size:(b + 1) * batch_size]
                if not batch: continue
                losses = self.train_step(batch)
                epoch_loss += losses["loss"]

            if n_batches > 0:
                epoch_loss /= n_batches

            # Early stopping condition
            if prev_loss is not None and abs(prev_loss - epoch_loss) < tol:
                break
            prev_loss = epoch_loss

        self.save_predictions_to_db() 
        
    def save_predictions_to_db(self):
        """
        Calculates predictions for all user-item pairs and performs 
        batch UPSERT to the database with memory-efficient JIT connections.
        """
        if self.db_config is None: 
            return
        
        batch_data = []
        batch_size = 2000 
        
        self.eval()
        with torch.no_grad():
            for u in self.users:
                for i in self.items:
                    try:
                        r_hat, _, _ = self.forward(u, i)
                        val = max(1.0, min(5.0, r_hat.item()))
                        batch_data.append((u, i, val))
                        
                        if len(batch_data) >= batch_size:
                            try:
                                with psycopg2.connect(**self.db_config) as conn:
                                    with conn.cursor() as cur:
                                        query = """
                                            INSERT INTO "Predict" ("UserId", "ItemId", "Value")
                                            VALUES %s
                                            ON CONFLICT ("UserId", "ItemId") DO UPDATE SET "Value" = EXCLUDED."Value";
                                        """
                                        psycopg2.extras.execute_values(cur, query, batch_data)
                                        conn.commit()
                            except Exception:
                                pass
                            batch_data = []
                    except Exception:
                        continue
            
            # Insert remaining predictions
            if batch_data:
                try:
                    with psycopg2.connect(**self.db_config) as conn:
                        with conn.cursor() as cur:
                            query = """
                                INSERT INTO "Predict" ("UserId", "ItemId", "Value")
                                VALUES %s
                                ON CONFLICT ("UserId", "ItemId") DO UPDATE SET "Value" = EXCLUDED."Value";
                            """
                            psycopg2.extras.execute_values(cur, query, batch_data)
                            conn.commit()
                except Exception:
                    pass
    
    def predict(self, user, item, p=0):
        """Inference function for a single prediction."""
        self.eval()
        with torch.no_grad():
            r_hat, _, _ = self.forward(user, item)
            return r_hat.item()

    def write_model_to_db(self):
        """Persists the learned ensemble weights (theta) to the database."""
        if self.db_config is None: return
        self.eval()

        theta_values = self.theta.detach().cpu().numpy() 
        model_order_names = [
            "ReviewRating", "UEIE", "UCInit", 
            "IInit 1", "IInit 2", "IInit 3", "IInit 4", 
            "IInit 5", "IInit 6", "IInit 7", "IInit 8"
        ]
        data_to_insert = []
        
        for idx, name in enumerate(model_order_names):
            mid = self.model_ids_map.get(name)
            if mid:
                data_to_insert.append((theta_values[idx].tolist(), mid))

        try:
            if data_to_insert:
                with psycopg2.connect(**self.db_config) as conn:
                    with conn.cursor() as cur:
                        query = 'UPDATE "Model" SET "LearnableParameters" = %s, "ModifiedAt" = NOW() WHERE "Id" = %s;'
                        psycopg2.extras.execute_batch(cur, query, data_to_insert)
                        conn.commit()
        except Exception:
            pass