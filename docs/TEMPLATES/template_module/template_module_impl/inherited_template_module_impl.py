# -*- coding: utf-8 -*-
from copy import deepcopy

from ..template_module_base import (TRACK_TEMPLATE_MODULES,
                                    VOS_TEMPLATE_MODULES, TemplateModuleBase)


@TRACK_TEMPLATE_MODULES.register
@VOS_TEMPLATE_MODULES.register
class InheritedTemplateModuleImplementation(TemplateModuleBase):
    r"""
    Template Module Implementation

    Hyper-parameters
    ----------------
    """
    extra_hyper_params = dict(inherited_template_module_hp="", )

    def __init__(self, ) -> None:
        super().__init__()

    def update_params(self) -> None:
        pass


InheritedTemplateModuleImplementation.default_hyper_params = deepcopy(
    InheritedTemplateModuleImplementation.default_hyper_params)
InheritedTemplateModuleImplementation.default_hyper_params.update(
    InheritedTemplateModuleImplementation.extra_hyper_params)
