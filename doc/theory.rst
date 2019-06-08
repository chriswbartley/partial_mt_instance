
Theory
========================

The monotone classification algorithm implemented here is described in the paper paper [bartley2017]_. A model is fit in two stages. First a standard `sci-kit learn` `RandomForestClassifier` is fit to the data, then the leaves are parsed to effectively remove non-monotone compliant leaves (i.e. those that do not comply with Theorem 3.1 and Lemma 4.1 in the paper). The result is a classifier that is perfectly monotone in the requested features, and very fast. Accuracy is not as good as later approaches by the same authors which use a different approach to monotonising the leaves.

.. DELETE_THIS_TO_USEmath::
    F(\textbf{x})=sign(a_0 + \sum_{m=1}^{M}a_m f_m(\textbf{x}))




.. [bartley2017] Bartley C., Liu W., Reynolds M. (2017). A Novel Framework for Partially Monotone Rule Ensembles. ICDE submission, prepub, http://staffhome.ecm.uwa.edu.au/~19514733/

