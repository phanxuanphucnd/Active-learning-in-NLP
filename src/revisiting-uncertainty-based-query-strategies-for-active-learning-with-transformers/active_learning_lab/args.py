import argparse

from argparse import Namespace
from jsonargparse import ArgumentParser

from active_learning_lab.utils.argparse import merge, namespace_to_dict


class KeywordArgs(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = dict()
        for kw_entry in values.split(','):
            key, value = kw_entry.strip().split('=')
            if key in kwargs:
                raise ValueError(f'Multiple assignments for argument \'{key}\'')
            else:
                kwargs[key] = value
        setattr(namespace, self.dest, kwargs)


class GroupingArgParser(ArgumentParser):

    def parse_args(self, args=None, namespace=None):
        cfg, _ = self._parse_known_args(args=args)
        self.check_config(cfg)

        args = super().parse_args(args=args, namespace=namespace)

        args_grouped = self._group_args(args)
        args_grouped = merge(self, args.config_files, args_grouped)

        args_grouped = namespace_to_dict(args_grouped)

        return args_grouped

    def parse_path(self, path, defaults=True):
        args = super().parse_path(path, defaults=defaults)
        args_grouped = self._group_args(args)

        return args_grouped

    def _group_args(self, args):

        args_grouped = Namespace()
        for group in self._action_groups:
            if group.title not in set(['positional arguments', 'optional arguments']):
                title_normalized = group.title.lower().replace(' ', '_')
                group_namespace = Namespace()

                for action in group._group_actions:
                    if hasattr(args, action.dest):
                        setattr(group_namespace, action.dest, getattr(args, action.dest))

                setattr(args_grouped, title_normalized, group_namespace)
            else:
                for action in group._group_actions:
                    if action.dest != 'help' and hasattr(args, action.dest):
                        setattr(args_grouped, action.dest, getattr(args, action.dest))
        return args_grouped


def get_parser():
    parser = GroupingArgParser(description='Runs an active learning experiment.')

    parser.add_argument('experiment_name', type=str, help='')
    parser.add_argument('config_files', nargs='+', default=[], help='')

    general_subcmd = parser.add_argument_group('general')
    general_subcmd.add_argument('--description', type=str, default='',
                                help='textual description or comments')
    general_subcmd.add_argument('--runs', type=int, default=5,
                                help='number of runs (repetitions)')
    general_subcmd.add_argument('--seed', type=int, default=1003, help='random seed')
    general_subcmd.add_argument('--max-reproducibility', action='store_true',
                                default=True, help='maximum reproducibility (slow)')

    ds_group = parser.add_argument_group('dataset')
    ds_group.add_argument('--dataset_name', type=str, help='name of dataset used')
    ds_group.add_argument('--dataset_kwargs', action=KeywordArgs, help='dataset keyword args')

    ds_group = parser.add_argument_group('classifier')
    ds_group.add_argument('--classifier_name', type=str, help='classifier to use')

    ds_group.add_argument('--validation_set_size', type=float,
                          default=0.1, help='size of the validation set '
                                            '(percentage of the train set)')
    ds_group.add_argument('--classifier_kwargs', action=KeywordArgs, help='classifier keyword args')

    al_group = parser.add_argument_group('active learner')
    al_group.add_argument('--num_queries', type=int, default=10,
                          help='number of active learning queries')
    al_group.add_argument('--query_size', type=int, default=100,
                          help='number of instances returned by a query step')

    al_group.add_argument('--initialization_strategy', type=str, default='random',
                          help='initialization to use')
    al_group.add_argument('--initialization_strategy_kwargs', action=KeywordArgs,
                          help='initialization keyword args')

    al_group.add_argument('--query_strategy', type=str, default='random',
                          help='query strategy to use')
    al_group.add_argument('--query_strategy_kwargs', action=KeywordArgs,
                          help='query strategy keyword args')

    return parser
