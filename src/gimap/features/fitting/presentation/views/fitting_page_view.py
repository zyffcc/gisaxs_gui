"""Hand-maintained Python View for the Fitting control surface."""

from PyQt5 import QtCore, QtWidgets

from .fitting_page_sections.page_shell import PageShellMixin
from .fitting_page_sections.input_section import InputSectionMixin
from .fitting_page_sections.fit_scaffold import FitScaffoldMixin
from .fitting_page_sections.particle_one import ParticleOneMixin
from .fitting_page_sections.particle_two import ParticleTwoMixin
from .fitting_page_sections.particle_three import ParticleThreeMixin
from .fitting_page_sections.particle_scroller import ParticleScrollerMixin
from .fitting_page_sections.fit_controls import FitControlsMixin
from .fitting_page_sections.page_finish import PageFinishMixin


class FittingPageView(
    PageShellMixin,
    InputSectionMixin,
    FitScaffoldMixin,
    ParticleOneMixin,
    ParticleTwoMixin,
    ParticleThreeMixin,
    ParticleScrollerMixin,
    FitControlsMixin,
    PageFinishMixin,
    object,
):
    def setupUi(self, gisaxsFittingPage):
        gisaxsFittingPage.setObjectName("gisaxsFittingPage")
        self._setup_page_shell(gisaxsFittingPage)
        self._setup_input_section()
        self._setup_fit_scaffold()
        self._setup_particle_one()
        self._setup_particle_two()
        self._setup_particle_three()
        self._finish_particle_scroller()
        self._setup_fit_controls()
        self._finish_page_shell(gisaxsFittingPage)

        self.retranslateUi(gisaxsFittingPage)
        self.fitParticleStackWidget_1.setCurrentIndex(1)
        self.fitParticleStackWidget_2.setCurrentIndex(0)
        self.fitParticleStackWidget_3.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(gisaxsFittingPage)

    def retranslateUi(self, gisaxsFittingPage):
        _translate = QtCore.QCoreApplication.translate
        self.gisaxsInputBox.setTitle(_translate("FittingPage", "GIMaP Input"))
        self.gisaxsInputCutLineLabel.setText(_translate("FittingPage", "Cut line:"))
        self.gisaxsInputColorScaleLabel.setText(_translate("FittingPage", "Color Scale:"))
        self.gisaxsInputImportButton.setText(_translate("FittingPage", "Import GISAXS"))
        self.gisaxsInputCenterAutoFindingButton.setText(_translate("FittingPage", "Auto Finding"))
        self.gisaxsInputVminLabel.setText(_translate("FittingPage", "Vmin:"))
        self.gisaxsInputStackValue.setText(_translate("FittingPage", "1"))
        self.gisaxsInputCenterVerticalLabel.setText(_translate("FittingPage", "Vertical."))
        self.gisaxsInputCenterParallelLabel.setText(_translate("FittingPage", "Parallel."))
        self.gisaxsInputAutoScaleCheckBox.setText(_translate("FittingPage", "Auto Scale"))
        self.gisaxsInputIntLogCheckBox.setText(_translate("FittingPage", "Int. Log"))
        self.gisaxsInputAutoShowCheckBox.setText(_translate("FittingPage", "Auto Show"))
        self.gisaxsInputShowButton.setText(_translate("FittingPage", "Show >"))
        self.gisaxsInputCutButton.setText(_translate("FittingPage", "Cut"))
        self.gisaxsInputVmaxLabel.setText(_translate("FittingPage", "Vmax:"))
        self.gisaxsInputCutLineVerticalLabel.setText(_translate("FittingPage", "Vertical."))
        self.gisaxsInputCutLineParallelLabel.setText(_translate("FittingPage", "Parallel."))
        self.gisaxsInputCenterLabel.setText(_translate("FittingPage", "Center:"))
        self.gisaxsInputDetectorParaButton.setText(_translate("FittingPage", "Detector Para."))
        self.gisaxsInputModelCombox.setItemText(0, _translate("FittingPage", "Single"))
        self.gisaxsInputModelCombox.setItemText(1, _translate("FittingPage", "Stack"))
        self.gisaxsInputModelCombox.setItemText(2, _translate("FittingPage", "In-situ"))
        self.fitBox.setTitle(_translate("FittingPage", "Fitting"))
        self.fitKLabel.setText(_translate("FittingPage", "k"))
        self.FittingManualFittingButton.setText(_translate("FittingPage", "Manual Fitting"))
        self.fitIntResLabel.setText(_translate("FittingPage", "Int [Res.]"))
        self.fitImport1dFileButton.setText(_translate("FittingPage", "Import 1D File"))
        self.fitCurrentDataCheckBox.setText(_translate("FittingPage", "GISAXS Data"))
        self.fitLogXCheckBox.setText(_translate("FittingPage", "Log-x"))
        self.fitLogYCheckBox.setText(_translate("FittingPage", "Log-y"))
        self.fitNormCheckBox.setText(_translate("FittingPage", "Norm"))
        self.FittingExportButton.setText(_translate("FittingPage", "Export"))
        self.fitParticleShapeCombox_1.setItemText(0, _translate("FittingPage", "Sphere"))
        self.fitParticleShapeCombox_1.setItemText(1, _translate("FittingPage", "Cylinder"))
        self.fitParticleShapeCombox_1.setItemText(2, _translate("FittingPage", "None"))
        self.fitParticleSphereDLabel_1.setText(_translate("FittingPage", "D [nm]"))
        self.fitParticleSphereSigmaDLabel_1.setText(_translate("FittingPage", "σ [D]"))
        self.fitParticleSphereRLabel_1.setText(_translate("FittingPage", "R [nm]"))
        self.fitParticleSphereBGLabel_1.setText(_translate("FittingPage", "BG"))
        self.fitParticleSphereSigmaRLabel_1.setText(_translate("FittingPage", "σ [R]"))
        self.fitParticleSphereIntLabel_1.setText(_translate("FittingPage", "Int."))
        self.fitParticleCylinderSigmaDLabel_1.setText(_translate("FittingPage", "σ [D]"))
        self.fitParticleCylinderRLabel_1.setText(_translate("FittingPage", "R [nm]"))
        self.fitParticleCylinderDLabel_1.setText(_translate("FittingPage", "D [nm]"))
        self.fitParticleCylinderIntLabel_1.setText(_translate("FittingPage", "Int."))
        self.fitParticleCylinderhLabel_1.setText(_translate("FittingPage", "h [nm]"))
        self.fitParticleCylinderSigmaRLabel_1.setText(_translate("FittingPage", "σ [R]"))
        self.fitParticleCylinderBGLabel_1.setText(_translate("FittingPage", "BG"))
        self.fitParticleCylinderSigmahLabel_1.setText(_translate("FittingPage", "σ [h]"))
        self.fitParticleShapeCombox_2.setItemText(0, _translate("FittingPage", "Sphere"))
        self.fitParticleShapeCombox_2.setItemText(1, _translate("FittingPage", "Cylinder"))
        self.fitParticleShapeCombox_2.setItemText(2, _translate("FittingPage", "None"))
        self.fitParticleSphereDLabel_2.setText(_translate("FittingPage", "D [nm]"))
        self.fitParticleSphereSigmaDLabel_2.setText(_translate("FittingPage", "σ [D]"))
        self.fitParticleSphereRLabel_2.setText(_translate("FittingPage", "R [nm]"))
        self.fitParticleSphereSigmaRLabel_2.setText(_translate("FittingPage", "σ [R]"))
        self.fitParticleSphereIntLabel_2.setText(_translate("FittingPage", "Int."))
        self.fitParticleSphereBGLabel_2.setText(_translate("FittingPage", "BG"))
        self.fitParticleCylinderRLabel_2.setText(_translate("FittingPage", "R [nm]"))
        self.fitParticleCylinderIntLabel_2.setText(_translate("FittingPage", "Int."))
        self.fitParticleCylinderSigmaDLabel_2.setText(_translate("FittingPage", "σ [D]"))
        self.fitParticleCylinderDLabel_2.setText(_translate("FittingPage", "D [nm]"))
        self.fitParticleCylinderSigmahLabel_2.setText(_translate("FittingPage", "σ [h]"))
        self.fitParticleCylinderSigmaRLabel_2.setText(_translate("FittingPage", "σ [R]"))
        self.fitParticleCylinderhLabel_2.setText(_translate("FittingPage", "h [nm]"))
        self.fitParticleCylinderBGLabel_2.setText(_translate("FittingPage", "BG"))
        self.fitParticleShapeCombox_3.setItemText(0, _translate("FittingPage", "Sphere"))
        self.fitParticleShapeCombox_3.setItemText(1, _translate("FittingPage", "Cylinder"))
        self.fitParticleShapeCombox_3.setItemText(2, _translate("FittingPage", "None"))
        self.fitParticleSphereRLabel_3.setText(_translate("FittingPage", "R [nm]"))
        self.fitParticleSphereIntLabel_3.setText(_translate("FittingPage", "Int."))
        self.fitParticleSphereSigmaDLabel_3.setText(_translate("FittingPage", "σ [D]"))
        self.fitParticleSphereSigmaRLabel_3.setText(_translate("FittingPage", "σ [R]"))
        self.fitParticleSphereDLabel_3.setText(_translate("FittingPage", "D [nm]"))
        self.fitParticleSphereBGLabel_3.setText(_translate("FittingPage", "BG"))
        self.fitParticleCylinderSigmaRLabel_3.setText(_translate("FittingPage", "σ [R]"))
        self.fitParticleCylinderBGLabel_3.setText(_translate("FittingPage", "BG"))
        self.fitParticleCylinderIntLabel_3.setText(_translate("FittingPage", "Int."))
        self.fitParticleCylinderRLabel_3.setText(_translate("FittingPage", "R [nm]"))
        self.fitParticleCylinderhLabel_3.setText(_translate("FittingPage", "h [nm]"))
        self.fitParticleCylinderSigmahLabel_3.setText(_translate("FittingPage", "σ [h]"))
        self.fitParticleCylinderSigmaDLabel_3.setText(_translate("FittingPage", "σ [D]"))
        self.fitParticleCylinderDLabel_3.setText(_translate("FittingPage", "D [nm]"))
        self.addModelButton.setText(_translate("FittingPage", "+"))
        self.fitInterpolationMethodLabel.setText(_translate("FittingPage", "Interpolation method:"))
        self.fitDataPointsNumLabel.setText(_translate("FittingPage", "Data Points Num.:"))
        self.fitFittingRegionLabel.setText(_translate("FittingPage", "Fitting Region"))
        self.fitBGShowCheckBox.setText(_translate("FittingPage", "BG"))
        self.fitDisplayOptionsLabel.setText(
            _translate("FittingPage", "Options displayed in the figure:")
        )
        self.fitResShowCheckBox.setText(_translate("FittingPage", "Res. Function"))
        self.fitParticle1ShowCheckBox.setText(_translate("FittingPage", "Particle 1"))
        self.fitParticle2ShowCheckBox.setText(_translate("FittingPage", "Particle 2"))
        self.fitParticle3ShowCheckBox.setText(_translate("FittingPage", "Particle 3"))
        self.fitMethodLabel.setText(_translate("FittingPage", "Method"))
        self.FittingAutoFittingButton.setText(_translate("FittingPage", "Auto Fitting"))
        self.fitMethodValue.setItemText(0, _translate("FittingPage", "Model: 1 Sphere"))
        self.fitMethodValue.setItemText(1, _translate("FittingPage", "Model: 2 Sphere"))
        self.fitMethodValue.setItemText(2, _translate("FittingPage", "Model: 3 Sphere"))
        self.fitMethodValue.setItemText(3, _translate("FittingPage", "Model: 1 Sphere + 1Cylinder"))
        self.fitMethodValue.setItemText(4, _translate("FittingPage", "Genetic Algorithm, GA"))
        self.FittingClearFittingButton_2.setText(_translate("FittingPage", "Clear Fitting"))
        self.FittingAutoKButton.setText(_translate("FittingPage", "<- Auto-K: OFF"))
        self.fitSigmaResLabel.setText(_translate("FittingPage", "σ [Res.]"))
        self.fitNuResLabel.setText(_translate("FittingPage", "v [Res.]"))
