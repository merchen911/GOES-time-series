# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



class BackboneSCINet(nn.Module):
    def __init__(
        self,
        n_out_steps,
        n_in_steps,
        n_in_features,
        d_hidden,
        n_stacks,
        n_levels,
        n_decoder_layers,
        n_groups,
        kernel_size=5,
        dropout: float = 0.5,
        concat_len: int = 0,
        pos_enc: bool = False,
        modified: bool = True,
        single_step_output_One: bool = False,
        output_bias = False
    ):
        super().__init__()

        self.n_in_steps = n_in_steps
        self.n_in_features = n_in_features
        self.n_out_steps = n_out_steps
        self.d_hidden = d_hidden
        self.n_levels = n_levels
        self.n_groups = n_groups
        self.modified = modified
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.concat_len = concat_len
        self.pos_enc = pos_enc
        self.single_step_output_One = single_step_output_One
        self.n_decoder_layers = n_decoder_layers
        self.output_bias = output_bias
        assert (
            self.n_in_steps % (np.power(2, self.n_levels)) == 0
        )  # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)

        self.blocks1 = EncoderTree(
            in_planes=self.n_in_features,
            num_levels=self.n_levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.n_groups,
            hidden_size=self.d_hidden,
            INN=modified,
        )

        if n_stacks == 2:  # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.n_in_features,
                num_levels=self.n_levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                groups=self.n_groups,
                hidden_size=self.d_hidden,
                INN=modified,
            )

        self.stacks = n_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.n_in_steps, self.n_out_steps, kernel_size=1, stride=1, bias=output_bias)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.n_in_steps // 4
        self.div_len = self.n_in_steps // 6

        if self.n_decoder_layers > 1:
            self.projection1 = nn.Linear(self.n_in_steps, self.n_out_steps)
            for layer_idx in range(self.n_decoder_layers - 1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i * self.div_len + self.overlap_len, self.n_in_steps) - i * self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        if self.single_step_output_One:  # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.n_out_steps, 1, kernel_size=1, bias=output_bias)
                else:
                    self.projection2 = nn.Conv1d(self.n_in_steps + self.n_out_steps, 1, kernel_size=1, bias=output_bias)
        else:  # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(
                        self.concat_len + self.n_out_steps,
                        self.n_out_steps,
                        kernel_size=1,
                        bias=False,
                    )
                else:
                    self.projection2 = nn.Conv1d(
                        self.n_in_steps + self.n_out_steps,
                        self.n_out_steps,
                        kernel_size=1,
                        bias=False,
                    )

        # For positional encoding
        self.pe_hidden_size = n_in_features
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1

        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        # temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.register_buffer("inv_timescales", inv_timescales)

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(
            max_length, dtype=torch.float32, device=x.device
        )  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        # temp1 = position.unsqueeze(1)  # 5 1
        # temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # [T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)

        return signal

    def forward(self, x):
        if self.pos_enc:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.n_decoder_layers == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0, 2, 1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape, dtype=x.dtype).to(x.device)
                for i, div_layer in enumerate(div_projection):
                    div_x = x[
                        :,
                        :,
                        i * self.div_len : min(i * self.div_len + self.overlap_len, self.n_in_steps),
                    ]
                    output[:, :, i * self.div_len : (i + 1) * self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0, 2, 1)

        if self.stacks == 1:
            return x, None

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len :, :], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            return x, MidOutPut





class Splitting(nn.Module):
    def __init__(self):
        super().__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        """Returns the odd and even part"""
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(
        self,
        in_planes,
        splitting=True,
        kernel=5,
        dropout=0.5,
        groups=1,
        hidden_size=1,
        INN=True,
    ):
        super().__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1  # by default: stride==1
            pad_r = self.dilation * (self.kernel_size) // 2 + 1  # by default: stride==1

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1  # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups, hidden_size, INN):
        super().__init__()
        self.level = Interactor(
            in_planes=in_planes,
            splitting=True,
            kernel=kernel,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)


class LevelSCINet(nn.Module):
    def __init__(self, in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.interact = InteractorLevel(
            in_planes=in_planes,
            kernel=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)  # even: B, T, D odd: B, T, D


class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.current_level = current_level

        self.workingblock = LevelSCINet(
            in_planes=in_planes,
            kernel_size=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

        if current_level != 0:
            self.SCINet_Tree_odd = SCINet_Tree(
                in_planes,
                current_level - 1,
                kernel_size,
                dropout,
                groups,
                hidden_size,
                INN,
            )
            self.SCINet_Tree_even = SCINet_Tree(
                in_planes,
                current_level - 1,
                kernel_size,
                dropout,
                groups,
                hidden_size,
                INN,
            )

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        # We recursively reordered these sub-series.
        # You can run the ./utils/recursive_demo.py to emulate this procedure.
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))


class EncoderTree(nn.Module):
    def __init__(self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels = num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes=in_planes,
            current_level=num_levels - 1,
            kernel_size=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

    def forward(self, x):
        x = self.SCINet_Tree(x)
        return x