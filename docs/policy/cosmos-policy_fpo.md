# 疑似コード: Cosmos-FPO 学習アルゴリズム

# --- 初期化 ---
# policy_net: Cosmos Policy (Flow Matchingに基づくActor, 潜在空間 z を入力)
# world_model: Cosmos Predictor (将来状態 z_t+1 を予測)
# critic: 価値関数 V(z_t)
# target_policy: 学習前の重みを固定した事前学習モデル (正則化用)

for epoch in range(max_epochs):
    # 1. データ収集 (Rollout)
    batch = collect_trajectories(policy_net) # (s_t, a_t, r_t, s_t+1)
    
    # 2. 潜在状態のエンコード (Cosmos Encoder)
    z_t = encoder(s_t)
    z_t_next = encoder(s_t+1)
    
    # 3. アドバンテージ計算
    # 報酬と価値関数に基づき、現在のアクションの良さを算出
    adv = r_t + gamma * critic(z_t_next) - critic(z_t)
    
    # 4. 単一目的の損失計算
    # --- World Model Loss ---
    # 世界モデルの物理予測誤差 (ELBOの一部)
    loss_world = mse(world_model(z_t, a_t), z_t_next)
    
    # --- Policy Optimization Loss (FPO++ base) ---
    # FPO++による方策勾配の近似。尤度計算をせず、ベクトル場の差分で更新
    # adv > 0 なら生成確率を高め、adv < 0 なら低める
    loss_policy = FPO_loss(policy_net, z_t, a_t, adv)
    
    # --- (Option) KL Regularization ---
    # 学習前のCosmosの「賢さ」を保持するためのオプション
    # 強力な推奨：これを入れないとRL中に物理法則を忘却する
    loss_kl = KL(policy_net(z_t) || target_policy(z_t))
    
    # 5. 合算した単一の目的関数 (Unified Objective)
    # alpha, beta, lambda は重みパラメータ
    total_loss = loss_policy + lambda_w * loss_world + beta * loss_kl + loss_critic
    
    # 6. バックプロパゲーションと重み更新
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 7. (Option) World Model Exploitation 防御
    # 世界モデルが「報酬獲得のためだけに歪む」のを防ぐため
    # 実際の観測 z_t+1 とのKL制約を強化する
    if option_enable_hallucination_control:
        apply_world_model_consistency_constraint()