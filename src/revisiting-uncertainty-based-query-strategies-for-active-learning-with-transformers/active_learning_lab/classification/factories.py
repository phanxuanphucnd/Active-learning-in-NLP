import copy

from small_text.classifiers.classification import ConfidenceEnhancedLinearSVC
from small_text.integrations.transformers.classifiers.classification import (
    FineTuningArguments,
    TransformerModelArguments
)

from active_learning_lab.classification.kimcnn_extension import KimCNNExtendedFactory
from active_learning_lab.classification.svm_extension import SklearnClassifierFactoryExtended
from active_learning_lab.classification.transformers_extension import (
    TransformerBasedClassificationExtendedFactory
)


def get_factory(classifier_name, num_classes, classifier_kwargs={}):

    if classifier_name == 'svm':
        return SklearnClassifierFactoryExtended(ConfidenceEnhancedLinearSVC(**classifier_kwargs))
    elif classifier_name == 'kimcnn':
        return KimCNNExtendedFactory(classifier_name, num_classes, kwargs=classifier_kwargs)
    elif classifier_name == 'transformer':
        kwargs_new = copy.deepcopy(classifier_kwargs)
        del kwargs_new['transformer_model']

        if 'scheduler' in classifier_kwargs and classifier_kwargs['scheduler'] == 'slanted':
            gradual_unfreezing = classifier_kwargs.get('gradual_unfreezing', -1)
            fine_tuning_args = FineTuningArguments(classifier_kwargs['lr'],
                                                   classifier_kwargs['layerwise_gradient_decay'],
                                                   gradual_unfreezing=gradual_unfreezing)
            kwargs_new['fine_tuning_arguments'] = fine_tuning_args

            if 'layerwise_gradient_decay' in kwargs_new:
                del kwargs_new['layerwise_gradient_decay']
            if 'gradual_unfreezing' in kwargs_new:
                del kwargs_new['gradual_unfreezing']

        transformer_model_args = TransformerModelArguments(classifier_kwargs['transformer_model'])
        return TransformerBasedClassificationExtendedFactory(transformer_model_args,
                                                             num_classes=num_classes,
                                                             kwargs=kwargs_new)

    raise RuntimeError(f'No factory available for classfier_name={classifier_name}')
