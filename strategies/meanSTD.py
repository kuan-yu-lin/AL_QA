

def mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
  		unlabeled_data,
        collate_fn=default_data_collator,
        batch_size=MODEL_BATCH,
    )
    print('Mean STD querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(n, len(unlabeled_data)))
    
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10).numpy()
    print('Got probability.')
    sigma_c = np.std(probs, axis=0)
    uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1)) # use tensor.sort() will sort the data and produce sorted indexes
    return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]