from src.preprocessing import load_data
from src.models import BetaVAE
from src.evaluation import perform_clustering, get_metrics
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Prepare Dataset
df = load_data('data/all_songs_data.csv', 'data/BanglaSongLyrics.csv')
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned']).toarray()

# 2. Train Model
model = BetaVAE(input_dim=X.shape[1])
model.compile(optimizer='adam')
model.fit(X, epochs=50, batch_size=32)

# 3. Cluster and Evaluate
latent = model.get_latent(X)
labels = perform_clustering(latent)
# (Optional: Add visualization code here)
print("Pipeline Finished Successfully.")
