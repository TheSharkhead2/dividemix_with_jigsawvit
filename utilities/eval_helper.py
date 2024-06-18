import os
import torch
import json
import pdb


class Eval_Helper:

    def __init__(self, train_folder, test_folder, threshold=3) -> None:
        # Config path should be a list of parent folders
        with open("config/classes.json", "r") as r:
            self.all_classes = json.load(r)
        self.missing_classes = []
        assert threshold >= 3, "Need to be greather than 2 to catch fake images"
        for key in self.all_classes.keys():
            # Verify that classes have at least 400 images
            # Missing classes only have 2
            if len(os.listdir(f"{train_folder}/train/{key}")) < threshold:
                self.missing_classes.append(key)
        with open(f"{test_folder}/missing_test_folders.json", "r") as r:
            additional_missing = json.load(r)
            for m in additional_missing:
                if m not in self.missing_classes:
                    self.missing_classes.append(m)
        self.missing_classes = sorted(self.missing_classes)
        self.missing_index = []
        for key in self.missing_classes:
            self.missing_index.append(self.all_classes[key])
        self.missing_index = torch.tensor(self.missing_index)

    def wrap(self, y_hat: torch.Tensor, y: torch.Tensor):
        self.missing_index = self.missing_index.to(y.device)  # type: ignore
        mask = ~torch.isin(y, self.missing_index)
        return y_hat[mask], y[mask]


# testing and sample code
if __name__ == "__main__":
    eval_helper = Eval_Helper("EAST", "EAST")
    print(f"Missing {len(eval_helper.missing_classes)} classes")