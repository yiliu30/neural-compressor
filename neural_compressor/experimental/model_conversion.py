#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helps convert one model format to another."""

import datetime
import tempfile

import yaml

from neural_compressor.adaptor import FRAMEWORKS

from ..conf.config import Conf
from ..conf.dotdict import DotDict, deep_get, deep_set
from ..model import BaseModel
from ..utils import logger
from ..utils.create_obj_from_config import create_dataloader, create_eval_func
from .common import Model as NCModel


class ModelConversion:  # pragma: no cover
    """ModelConversion class is used to convert one model format to another.

       Currently Neural Compressor only supports Quantization-aware training TensorFlow model to Default
       quantized model.

       The typical usage is:
         from neural_compressor.experimental import ModelConversion, common
         conversion = ModelConversion()
         conversion.source = 'QAT'
         conversion.destination = 'default'
         conversion.model = '/path/to/saved_model'
         q_model = conversion()

    Args:
        conf_fname_or_obj (string or obj): Optional. The path to the YAML configuration file or
            Conf class containing model conversion and evaluation setting if not specifed by code.
    """

    def __init__(self, conf_fname_or_obj=None):
        """Initialize the ModelConversion class.

        Args:
            conf_fname_or_obj (string or obj): Optional. The path to the YAML configuration file or
                Conf class containing model conversion and evaluation setting if not specifed by code.
        """
        self.conf_name = conf_fname_or_obj
        self._model = None
        self.framework = "tensorflow"

        self._eval_dataloader = None
        self._eval_func = None
        self.adaptor = None
        self._metric = None

        self._source = None
        self._destination = None

        if conf_fname_or_obj is not None:
            if isinstance(conf_fname_or_obj, str):
                self.conf = Conf(conf_fname_or_obj)
            elif isinstance(conf_fname_or_obj, Conf):
                self.conf = conf_fname_or_obj
            else:  # pragma: no cover
                assert (
                    False
                ), "Please pass a YAML configuration file path or \
                    Conf class to model_conversion"
        else:
            self.conf = None

    def __call__(self):
        """Execute model conversion process.

           NOTE: This interface works now only on Intel Optimized TensorFlow to
           convert QAT model generated by tensorflow_model_optimization to default
           quantized model which is able to run at Intel Xeon platforms.
           https://github.com/tensorflow/model-optimization

        Returns:
            converted quantized model
        """
        assert self._model, '"model" property need to be set before __call_() gets invoked'

        framework_specific_info = {}
        cfg = self.conf.usr_cfg
        framework_specific_info.update(
            {
                "name": cfg.model.name,
                "backend": "default",
                "format": "default",
                "device": cfg.device,
                "fake_quant": True,
                "inputs": cfg.model.inputs,
                "outputs": cfg.model.outputs,
                "workspace_path": cfg.tuning.workspace.path,
            }
        )

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)
        q_model = self.adaptor.convert(self._model, self._source, self._destination)

        # when eval_func is None but metric or _eval_dataloader is set by yaml or code,
        # it means Neural Compressor will create the eval_func from these info.
        metric_cfg = [self._metric] if self._metric else deep_get(cfg, "evaluation.accuracy.metric")
        postprocess_cfg = deep_get(cfg, "evaluation.accuracy.postprocess")
        if self._eval_func is None and metric_cfg:
            eval_dataloader_cfg = deep_get(cfg, "evaluation.accuracy.dataloader")
            if self._eval_dataloader is None and eval_dataloader_cfg:
                self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)
            assert self._eval_dataloader, (
                'either "eval_dataloader" property or evaluation'
                ".accuracy.dataloader field in yaml should be set when metric is set"
            )

            self._eval_func = create_eval_func(
                self.framework, self.eval_dataloader, self.adaptor, metric_cfg, postprocess_cfg, fp32_baseline=True
            )
        if self._eval_func:
            baseline_score = self._eval_func(self._model)
            qmodel_score = self._eval_func(q_model)
            logger.info("The score of Quantization-Aware Training model is {}.".format(str(baseline_score)))
            logger.info("Converted model score is {}.".format(str(qmodel_score)))

        return q_model

    fit = __call__

    def _gen_yaml(self):
        random_name = "{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        default_yaml_template = {
            "model": {"framework": self.framework, "name": random_name},
            "device": "cpu",
            "model_conversion": {"source": "QAT", "destination": "default"},
        }

        temp_yaml_path = tempfile.mkstemp(suffix=".yaml")[1]
        with open(temp_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(default_yaml_template, f)
        self.conf = Conf(temp_yaml_path)

    def dataset(self, dataset_type, *args, **kwargs):
        """Return dataset.

        Args:
            dataset_type: dataset type

        Returns:
            class: dataset class
        """
        from .data import Datasets

        return Datasets(self.framework)[dataset_type](*args, **kwargs)

    @property
    def source(self):
        """Return source."""
        return self._source

    @source.setter
    def source(self, _source):
        """Set source."""
        assert _source.lower() == "qat", (
            "Model conversion now only supports TensorFlow " "QAT model to default quantized model"
        )
        self._source = _source.lower()

    @property
    def destination(self):
        """Return destination."""
        return self._destination

    @destination.setter
    def destination(self, _destination):
        """Set destination."""
        assert _destination.lower() == "default", (
            "Model conversion now only supports " "TensorFlow QAT model to default quantized model"
        )
        self._destination = _destination.lower()

    @property
    def eval_dataloader(self):
        """Return eval dataloader."""
        return self._eval_dataloader

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        """Set Data loader for evaluation.

        It is iterable and the batched data should consists of a tuple like (input, label),
        when eval_dataloader is set, user should configure postprocess(optional) and metric
        in yaml file or set postprocess and metric cls. Notice evaluation dataloader will be
        used to generate data for model inference, make sure the input data can be feed to model.

        Args:
            dataloader(generator): user are supported to set a user defined dataloader
                                    which meet the requirements that can yield tuple of
                                    (input, label)/(input, _) batched data.
                                    Another good practice is to use neural_compressor.common.DataLoader
                                    to initialize a neural_compressor dataloader object.
                                    Notice neural_compressor.common.DataLoader is just a wrapper of the
                                    information needed to build a dataloader, it can't yield
                                    batched data and only in this setter method
                                    a 'real' eval_dataloader will be created,
                                    the reason is we have to know the framework info
                                    and only after the Quantization object created then
                                    framework infomation can be known. Future we will support
                                    creating iterable dataloader from neural_compressor.common.DataLoader
        """
        from .common import _generate_common_dataloader

        self._eval_dataloader = _generate_common_dataloader(dataloader, self.framework)

    @property
    def model(self):
        """Return model."""
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object.

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model), but not recommended.
                       Best practice is to set from a initialized neural_compressor.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be auto inferred,
                       but sometimes auto inferred inputs/outputs will not meet your requests,
                       set them manually in config yaml file. Another corner case is slim model
                       of tensorflow, be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.
        """
        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            self._model = NCModel(user_model)
        else:
            self._model = user_model

        assert self.framework == "tensorflow", "Model conversion only supports Tensorflow at current stage."

        if not self.conf:
            self._gen_yaml()

        cfg = self.conf.usr_cfg
        if self.framework == "tensorflow":
            self._model.name = cfg.model.name
            self._model.workspace_path = cfg.tuning.workspace.path

    @property
    def metric(self):
        """Return metric."""
        assert False, "Should not try to get the value of `metric` attribute."
        return None

    @metric.setter
    def metric(self, user_metric):
        """Set the metric class.

        Set metric class and neural_compressor will initialize this class when evaluation
        neural_compressor have many built-in metrics, but user can set specific metric through
        this api. The metric class should take the outputs of the model or
        postprocess(if have) as inputs, neural_compressor built-in metric always take
        (predictions, labels) as inputs for update,
        and user_metric.metric_cls should be sub_class of neural_compressor.metric.BaseMetric
        or user defined metric object.

        Args:
            user_metric(neural_compressor.common.Metric): user_metric should be object initialized from
                                             neural_compressor.common.Metric, in this method the
                                             user_metric.metric_cls will be registered to
                                             specific frameworks and initialized.
        """
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning(
                "Override the value of `metric` field defined in yaml file"
                " as user defines the value of `metric` attribute by code."
            )

        from .common import Metric as NCMetric

        if isinstance(user_metric, NCMetric):
            metric_cfg = {user_metric.name: {**user_metric.kwargs}}
            deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
            self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
            from .metric import METRICS

            metrics = METRICS(self.framework)
            metrics.register(user_metric.name, user_metric.metric_cls)
        else:
            for i in ["reset", "update", "result"]:
                assert hasattr(user_metric, i), "Please realise {} function" "in user defined metric".format(i)
            self._metric = user_metric

    @property
    def postprocess(self):
        """Check postprocess."""
        assert False, "Should not try to get the value of `postprocess` attribute."
        return None

    @postprocess.setter
    def postprocess(self, user_postprocess):
        """Set postprocess class and neural_compressor will initialize this class when evaluation.

        The postprocess class should take the outputs of the model as inputs, and
        output (predictions, labels) as inputs for metric update.
        user_postprocess.postprocess_cls should be sub_class of neural_compressor.data.BaseTransform.

        Args:
            user_postprocess(neural_compressor.common.Postprocess):user_postprocess should be
                            object initialized from neural_compressor.common.Postprocess,
                            in this method the user_postprocess.postprocess_cls will be
                            registered to specific frameworks and initialized.
        """
        from .common import Postprocess as NCPostprocess

        assert isinstance(
            user_postprocess, NCPostprocess
        ), "please initialize a neural_compressor.common.Postprocess and set...."
        postprocess_cfg = {user_postprocess.name: {**user_postprocess.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.postprocess"):
            logger.warning(
                "Override the value of `postprocess` field defined in yaml file"
                " as user defines the value of `postprocess` attribute by code."
            )
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        from .data import TRANSFORMS

        postprocesses = TRANSFORMS(self.framework, "postprocess")
        postprocesses.register(user_postprocess.name, user_postprocess.postprocess_cls)

    @property
    def eval_func(self):
        """Return eval_func."""
        assert False, "Should not try to get the value of `eval_func` attribute."
        return None

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """Set the evaluation function provided by user.

        Args:
            user_eval_func: This function takes model as parameter,
                            and evaluation dataset and metrics should be
                            encapsulated in this function implementation
                            and outputs a higher-is-better accuracy scalar
                            value.
        """
        self._eval_func = user_eval_func

    def __repr__(self):
        """Return representation."""
        return "ModelConversion"
