import numpy as np

from sklearn.utils.validation import check_scalar, column_or_1d, check_random_state, check_consistent_length


def cross_validation(sample_indices, n_folds=5, random_state=None, y=None):
    """
    Performs a (stratified) k-fold cross-validation.

    Parameters
    ----------
    sample_indices : int
        Array of sample indices.
    n_folds : int, default=5
        Number of folds. Must be at least 2.
    random_state : int, RandomState instance or None, default=None
        `random_state` affects the ordering of the indices, which controls the randomness of each fold.

    Returns
    -------
    train : list
        Contains the training indices of each iteration, where train[i] represents iteration i.
    test : list
        Contains the test indices of each iteration, where test[i] represents iteration i.
    """
    # aus sample_indices die train bzw. test indices enthalten
    # Elemente der Liste sollen einzelne Folds repräsentieren

    # checks and balances
    sample_indices = column_or_1d(sample_indices, dtype=int).copy()
    n_folds = check_scalar(n_folds, 'n_folds', target_type=int, min_val=2, max_val=len(sample_indices))

    # random state
    random_state = check_random_state(random_state) # returns random state if random_state is None

    # stratification check
    y = column_or_1d(y) if y is not None else np.zeros(len(sample_indices))
    check_consistent_length(sample_indices, y)
    classes, counts = np.unique(y, return_counts=True)
    classes = classes[np.argsort(-counts)]  # argsort: von größten zum kleinsten Element -> umdrehen durch -1

    # data shuffling
    p = random_state.permutation(len(sample_indices)) # random permutation of indices
    sample_indices = sample_indices[p] # shuffle sample indices
    y = y[p] # shuffle y

    # initialize variables (return variables)
    folds = [[] for _ in range(n_folds)] # initialize empty list of lists: [] for every fold
    fold_indices = np.arange(n_folds)
    train, test = [], []

    # fold befüllen
    # stratified k-fold cross-validation: samples are drawn in a way that preserves the percentage of samples for each class
    # -> performace measure auf Test Daten repränsentiert die gesamte performance des Modells
    # -> auf repräsentativen Daten trainieren

    # wir haben schon permuted -> y splitten nach Klasse und Anteil pro Klasse in fold einfügen
    for class_y in classes: # von most zu least populated class
        is_class_y = y == class_y # indices of class_y
        folds_class_y = list(np.array_split(sample_indices[is_class_y], n_folds)) # split member of class y into n_folds: Jedes Element ist in genau einem Fold
        fold_lengths = [len(fold) for fold in folds] # lengths of folds, erste Iteration 0
        sort_idx = np.argsort(fold_lengths) # sort indices of fold_lengths (kleinste zu größte Länge)
        for f_y, f in enumerate(fold_indices[sort_idx]): # f_y: fold index, f: index of fold
            folds[f].extend(folds_class_y[f_y]) # extend fold f with samples of class y

    # folds befüllt, jetzt train und test indices erstellen
    for f_1 in fold_indices:
        test.append(folds[f_1]) # immer in Fold als test set, Liste aus Listen
        train_f_1 = []  # einen train fold erstellen
        for f_2 in fold_indices:
            if f_1 != f_2:
                train_f_1.extend(folds[f_2])    # alle Elemente außer f_1 in train, nicht appenden, damit nicht zu viele Listen Strukturen entstehen
        train.append(train_f_1) # alle train sets (aller folds)
    return train, test

