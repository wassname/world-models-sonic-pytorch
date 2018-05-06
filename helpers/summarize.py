# summarize model
from collections import OrderedDict
import pandas as pd

import torch
from torch import nn


def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=''):
        for key, module in module.named_children():
            name = parent_name + '.' + key if parent_name else key
            names[name] = module
            if isinstance(module, torch.nn.Module):
                _get_names(module, parent_name=name)
    _get_names(model)
    return names


class TorchSummarizeDf(object):
    def __init__(self, model, weights=False, input_shape=True, nb_trainable=False, debug=False):
        """
        Summarizes torch model by showing trainable parameters and weights.

        author: wassname
        url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
        license: MIT

        Modified from:
        - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
        - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/

        Usage:
            import torchvision.models as models
            model = models.alexnet()
            # attach temporary hooks using `with`
            with TorchSummarizeDf(model) as tdf:
                x = Variable(torch.rand(2, 3, 224, 224))
                y = model(x)
                df = tdf.make_df()
            print(df)

            # Total parameters 61100840
            #              name class_name        input_shape       output_shape  nb_params
            # 1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296
            # 2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
            # ...
        """
        # Names are stored in parent and path+name is unique not the name
        self.names = get_names_dict(model)

        # store arguments
        self.model = model
        self.weights = weights
        self.input_shape = input_shape
        self.nb_trainable = nb_trainable
        self.debug = debug

        # create properties
        self.summary = OrderedDict()
        self.hooks = []

    def register_hook(self, module):
        """Register hooks recursively"""

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == self.model):
            self.hooks.append(module.register_forward_hook(self.hook))

    def hook(self, module, input, output):
        """This hook is applied when each module is run"""
        name = ''
        for key, item in self.names.items():
            if item == module:
                name = key

        class_name = str(module.__class__).split('.')[-1].split("'")[0]
        module_idx = len(self.summary)

        m_key = module_idx + 1

        self.summary[m_key] = OrderedDict()
        self.summary[m_key]['name'] = name
        self.summary[m_key]['class_name'] = class_name

        # Handle multiple inputs
        if not isinstance(input, (list, tuple)):
            input = (input, )
        if not isinstance(input[0], (list, tuple)):
            input = (input, )
        if self.input_shape:
            # for each input remove batch size and replace with one
            input_size = [[(-1, ) + tuple(oo.size()[1:]) for oo in o] for o in input if o is not None]
            self.summary[m_key][
                'input_shape'] = input_size

        # Handle multiple outputs
        if not isinstance(output, (list, tuple)):
            output = (output, )
        if not isinstance(output[0], (list, tuple)):
            output = (output, )

        output_size = [[(-1, ) + tuple(oo.size()[1:]) for oo in o] for o in output if o is not None]
        self.summary[m_key]['output_shape'] = output_size

        if self.weights:
            self.summary[m_key]['weights'] = list(
                [tuple(p.size()) for p in module.parameters()])

            # TODO filter to trainable params?
    #             summary[m_key]['trainable'] = any([p.requires_grad for p in module.parameters()])

        if self.nb_trainable:
            params_trainable = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])
            self.summary[m_key]['nb_trainable'] = params_trainable
        params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])
        self.summary[m_key]['nb_params'] = params
        if self.debug:
            print(self.summary[m_key])

    def __enter__(self):

        # register hook
        self.model.apply(self.register_hook)

        # make a forward pass
        self.training = self.model.training
        if self.training:
            self.model.eval()

        return self

    def make_df(self):
        """Make dataframe."""
        df = pd.DataFrame.from_dict(self.summary, orient='index')

        df['level'] = df['name'].apply(lambda name: name.count('.'))
        total_toplevel_params = df[df['level'] == df['level'].min()]['nb_params'].sum()
        print('Total parameters', total_toplevel_params)

        return df

    def __exit__(self, exc_type, exc_val, exc_tb):

        if exc_type or exc_val or exc_tb:
            # to help with debugging your model lets print the summary even if it fails
            df_summary = pd.DataFrame.from_dict(self.summary, orient='index')
            print(df_summary)

        if self.training:
            self.model.train()

        # remove these hooks
        for h in self.hooks:
            h.remove()
