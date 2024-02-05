import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm


class _Unifier:
    @staticmethod
    def train(datahub, set_type, extractor=None, inter_func=None, **kwargs):
        if isinstance(inter_func, nn.Module):
            dataloader = datahub.to_dataloader(
                batch_size=kwargs["batch_size"],
                dtype=extractor.dtype,
                set_type=set_type,
                label=True
            )
            loss_func = kwargs["loss_func"]
            optimizer = kwargs["optimizer"]
            device = extractor.device
            epoch_losses = []
            extractor.train()
            inter_func.train()
            for batch_data in tqdm(dataloader, "Training"):
                student_id, exercise_id, q_mask, r = batch_data
                student_id: torch.Tensor = student_id.to(device)
                exercise_id: torch.Tensor = exercise_id.to(device)
                q_mask: torch.Tensor = q_mask.to(device)
                r: torch.Tensor = r.to(device)
                _ = extractor.extract(student_id, exercise_id, q_mask)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                compute_params = {
                    'student_ts': student_ts,
                    'diff_ts': diff_ts,
                    'disc_ts': disc_ts,
                    'q_mask': q_mask,
                    'knowledge_ts': knowledge_ts,
                }

                if len(_) > 4:
                    compute_params['other'] = _[4]
                    if isinstance(_[4], dict):
                        extra_loss = _[4].get('extra_loss', 0)

                else:
                    extra_loss = 0
                pred_r: torch.Tensor = inter_func.compute(**compute_params)
                loss = loss_func(pred_r, r) + extra_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                inter_func.monotonicity()
                epoch_losses.append(loss.mean().item())
            print("Average loss: {}".format(float(np.mean(epoch_losses))))
        # To cope with statistics methods
        else:
            ...

    @staticmethod
    def predict(datahub, set_type, extractor=None, inter_func=None, **kwargs):
        if isinstance(inter_func, nn.Module):
            dataloader = datahub.to_dataloader(
                batch_size=kwargs["batch_size"],
                dtype=extractor.dtype,
                set_type=set_type,
                label=False
            )
            device = extractor.device
            extractor.eval()
            inter_func.eval()
            pred = []
            for batch_data in tqdm(dataloader, "Evaluating"):
                student_id, exercise_id, q_mask = batch_data
                student_id: torch.Tensor = student_id.to(device)
                exercise_id: torch.Tensor = exercise_id.to(device)
                q_mask: torch.Tensor = q_mask.to(device)
                _ = extractor.extract(student_id, exercise_id, q_mask)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                compute_params = {
                    'student_ts': student_ts,
                    'diff_ts': diff_ts,
                    'disc_ts': disc_ts,
                    'q_mask': q_mask,
                    'knowledge_ts': knowledge_ts
                }
                if len(_) > 4:
                    compute_params['other'] = _[4]
                pred_r: torch.Tensor = inter_func.compute(**compute_params)
                pred.extend(pred_r.detach().cpu().tolist())
            return pred
        # To cope with statistics methods
        else:
            ...
