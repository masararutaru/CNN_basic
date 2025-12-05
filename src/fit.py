def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, history):
    import torch
    import numpy as np
    from tqdm import tqdm
    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs + base_epochs):
        #1エポックあたりの正解数
        n_train_acc, n_val_acc = 0, 0
        #1エポック当たりの累積損失（平均化前）
        train_loss, val_loss = 0, 0
        #1エポック当たりの累積データ件数
        n_train, n_test = 0, 0
        #訓練フェーズ

        for inputs, labels in tqdm(train_loader):
            #1バッチ当たりのデータ件数
            train_batch_size = len(labels)
            #1エポックあたりのデータ累積件数
            n_train += train_batch_size

            #勾配初期化
            optimizer.zero_grad()
            #予測計算
            outputs = net(inputs)
            #損失計算
            loss = criterion(outputs, labels)
            #勾配計算
            loss.backward()
            #パラメータ更新
            optimizer.step()
            #予測ラベル導出
            predicted = torch.max(outputs, 1)[1]
            #平均前の損失と正解数の計算
            #lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()
        
        net.eval()

        for inputs_test, labels_test in test_loader:
            #1バッチ当たりのデータ件数
            test_batch_size = len(labels_test)
            #1エポック当たりのデータ累積件数
            n_test += test_batch_size

            #予測計算
            outputs_test = net(inputs_test)

            #損失計算
            loss_test = criterion(outputs_test, labels_test)

            #予測ラベル導出
            predicted_test = torch.max(outputs_test, 1)[1]

            #平均前の損失と正解数の計算
            #lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss += loss_test.item() * test_batch_size
            n_val_acc += (predicted_test == labels_test).sum().item()

        #精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        #損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        #結果表示
        print(f'Epoch[{epoch+1}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
        #記録
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))
        

    return history
