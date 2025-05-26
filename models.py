from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge

def get_models():
    return {
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        # "KNN": KNeighborsRegressor(n_neighbors=5),
        "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42),
        "Lasso": Lasso(alpha=0.1),
        # "Elastic_Net":ElasticNet(alpha=1.0, l1_ratio=0.5),
        # "BayesianRidge":BayesianRidge(),
        }
