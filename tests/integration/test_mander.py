import base64

# import watermark
import io
import time

import numpy as np
from mandr import InfoMander
from mandr.templates import TemplateRenderer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


def test_mandr(tmp_path):
    def train_model():
        X, y = make_classification(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        tic = time.time()
        clf = LogisticRegression(random_state=0)
        grid = GridSearchCV(
            clf, param_grid={"C": np.logspace(0.0001, 1, 10), "random_state": range(10)}
        )
        grid.fit(X_train, y_train)
        toc = time.time()
        return grid, tic, toc

    for i in range(5):
        X, y = make_classification(random_state=i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        mander = InfoMander(f"probabl-ai/demo-usecase/training/{i}", root=tmp_path)

        grid, tic, toc = train_model()

        template = """
        This dataset has {{n_classes}} classes.
    
        {{ scatter_chart(title="my chart", x=mander.cv_results.mean_test_score, y=mander.cv_results.std_test_score) }}
        """

        cm = confusion_matrix(y_test, grid.predict(X_test), labels=grid.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)

        # Save a matplotlib figure
        myio = io.BytesIO()
        disp.savefig(myio, format="jpg")
        myio.seek(0)
        base64_img = base64.b64encode(myio.read()).decode()
        # Make it possible to visualize it from the "views" menu
        mander.add_view(
            "confusion_matrix", f'<img src="data:image/png;base64, {base64_img}">'
        )

        # mander.add_logs('watermark', watermark.watermark().replace('\n','<br>'))
        mander.add_artifact("model", grid)
        mander.add_info("train_time", toc - tic)
        mander.add_info("cv_results", grid.cv_results_)
        mander.add_info("X_shape", list(X.shape))
        mander.add_info("n_classes", len(set(y)))
        mander.add_view("index", TemplateRenderer(mander).render(template))
        # mander.add_logs('watermark', watermark.watermark().replace('\n','<br>'))

        assert "This dataset has 2 classes." in mander["views"]["index"]
        assert '"title": "my chart"' in mander["views"]["index"]
