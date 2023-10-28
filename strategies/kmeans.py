

def kmeans_query(n_pool, labeled_idxs, train_dataset, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    # unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      collate_fn=default_data_collator,
                                      batch_size=MODEL_BATCH,
                                    )
    print('KMean querying starts.')
    print('Query {} data.'.format(n))
    
    embeddings = get_embeddings(unlabeled_dataloader, device)
    print('Got embeddings.')
    embeddings = embeddings.numpy()

    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(embeddings)
    cluster_idxs = cluster_learner.predict(embeddings)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeddings - centers)**2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

    return unlabeled_idxs[q_idxs]