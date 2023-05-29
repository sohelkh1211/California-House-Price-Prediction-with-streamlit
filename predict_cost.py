from house import model
import joblib
joblib.dump(model, "model.sav")
def predict(data):
  return model.predict(data)
