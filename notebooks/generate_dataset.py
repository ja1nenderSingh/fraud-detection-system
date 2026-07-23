import numpy as np
import pandas as pd

# Set seed so dataset is consistent every time
np.random.seed(42)

n_normal = 28000
n_fraud = 492

# Generate normal transactions
normal = pd.DataFrame({
    'Time': np.random.randint(0, 172800, n_normal),
    'Amount': np.random.exponential(50, n_normal),
    **{f'V{i}': np.random.normal(0, 1, n_normal) for i in range(1, 29)},
    'Class': 0
})

# Generate fraud transactions (different pattern)
fraud = pd.DataFrame({
    'Time': np.random.randint(0, 172800, n_fraud),
    'Amount': np.random.exponential(200, n_fraud),
    **{f'V{i}': np.random.normal(2, 3, n_fraud) for i in range(1, 29)},
    'Class': 1
})

# Combine and shuffle
df = pd.concat([normal, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to data folder
df.to_csv('../data/creditcard.csv', index=False)

print(f"Dataset created successfully!")
print(f"Total transactions: {len(df):,}")
print(f"Normal transactions: {len(normal):,}")
print(f"Fraud transactions: {len(fraud):,}")
print(f"Fraud percentage: {len(fraud)/len(df)*100:.2f}%")
print(f"Saved to data/creditcard.csv")