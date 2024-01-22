import torch


class DeepTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def fit(self, train_loader, epochs, print_every_n_epochs=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                loss = self._train_batch(batch)
                total_loss += loss

            if epoch % print_every_n_epochs == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    def _train_batch(self, batch):
        inputs, targets = batch
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, targets).item()
        return total_loss / len(test_loader)
