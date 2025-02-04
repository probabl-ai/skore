def test_comparator():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from skore import EstimatorReport

    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LogisticRegression().fit(X_train, y_train)

    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    from skore import Comparator

    comp = Comparator([report, report])

    comp.metrics.help()
    comp.metrics.report_metrics()
    comp.metrics.brier_score()
    comp.metrics.plot.roc()
    print(comp.metrics.accuracy())
    print(comp.metrics.accuracy(aggregate="mean"))
    print(comp.metrics.accuracy(aggregate=["mean", "std"]))


def test_comparator_different_estimators():
    from sklearn.datasets import make_classification
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from skore import EstimatorReport

    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    logistic_regression = LogisticRegression().fit(X_train, y_train)
    dummy_classifier = DummyClassifier().fit(X_train, y_train)

    from skore import Comparator

    comp = Comparator(
        [
            EstimatorReport(logistic_regression, X_test=X_test, y_test=y_test),
            EstimatorReport(dummy_classifier, X_test=X_test, y_test=y_test),
        ]
    )

    comp.metrics.help()
    comp.metrics.report_metrics()
    comp.metrics.brier_score()
    comp.metrics.plot.roc()
    print(comp.metrics.accuracy())
    print(comp.metrics.accuracy(aggregate="mean"))
    print(comp.metrics.accuracy(aggregate=["mean", "std"]))


def test_example(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from skore import Comparator, EstimatorReport, Project, train_test_split

    X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    log_reg = LogisticRegression()
    est_report_lr = EstimatorReport(
        log_reg, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    est_report_rf = EstimatorReport(
        rf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    project = Project(tmp_path / "my-project.skore")
    project.put("est_rep", est_report_lr)
    project.put("est_rep", est_report_rf)

    comp1 = Comparator([est_report_lr, est_report_rf])
    comp2 = Comparator(project.get("est_rep", version="all"))

    print(comp1.metrics.accuracy())
    print(comp2.metrics.accuracy())
