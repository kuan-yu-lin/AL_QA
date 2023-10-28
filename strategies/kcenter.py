

def kcenter_greedy_query(n_pool, labeled_idxs, train_dataset, device, n):
    labeled_idxs_in_query = labeled_idxs.copy()
    # train_data = train_dataset
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  batch_size=MODEL_BATCH,
                                )
    print('KCenter greedy querying starts.')
    print('Query {} data.'.format(n))
    
    embeddings = get_embeddings(train_dataloader, device)
    print('Got embeddings.')
    embeddings = embeddings.numpy()

    dist_mat = np.matmul(embeddings, embeddings.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs_in_query), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    mat = dist_mat[~labeled_idxs_in_query, :][:, labeled_idxs_in_query]

    for i in tqdm(range(n), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(n_pool)[~labeled_idxs_in_query][q_idx_]
        labeled_idxs_in_query[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_idxs_in_query, q_idx][:, None], axis=1)
        
    return np.arange(n_pool)[(labeled_idxs ^ labeled_idxs_in_query)]

def kcenter_greedy_PCA_query(n_pool, labeled_idxs, train_dataset, device, n):
    labeled_idxs_in_query = labeled_idxs.copy()
    # train_data = train_dataset
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  batch_size=MODEL_BATCH,
                                )
    print('KCenter greedy PCA querying starts.')
    print('Query {} data.'.format(n))

    embeddings = get_embeddings(train_dataloader, device)
    print('Got embeddings.')
    embeddings = embeddings.numpy()
    dist_mat = np.matmul(embeddings, embeddings.transpose())

    if len(embeddings[0]) > 50:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)
    embeddings = embeddings.astype(np.float16)

    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs_in_query), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    mat = dist_mat[~labeled_idxs_in_query, :][:, labeled_idxs_in_query]

    for i in tqdm(range(n), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(n_pool)[~labeled_idxs_in_query][q_idx_]
        labeled_idxs_in_query[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_idxs_in_query, q_idx][:, None], axis=1)
        
    return np.arange(n_pool)[(labeled_idxs ^ labeled_idxs_in_query)]