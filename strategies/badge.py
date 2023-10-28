

def badge_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      collate_fn=default_data_collator,
                                      batch_size=MODEL_BATCH,
                                    )
    print('BADGE querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(n, len(unlabeled_data)))

    gradEmbedding = get_grad_embeddings(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got embeddings.')
    chosen = init_centers(gradEmbedding, n)
    return unlabeled_idxs[chosen]