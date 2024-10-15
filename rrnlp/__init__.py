'''
A module that ties together the constituent models into a single
interface; will pull everything it can from an input article.

Note: if you find this useful, please see:
        https://github.com/bwallace/RRnlp#citation.
'''
from typing import Dict, List, Tuple, Type

import copy
import warnings

import rrnlp

from .models import (
    RCT_classifier,
    PICO_tagger,
    ev_inf_classifier,
    ICOev_extractor,
    sample_size_extractor,
    RoB_classifier_LR,
    study_design_classifier,
    get_device,
    SearchBot,
    ScreenerBot,
    NumericalExtractionBot,
    MultiSummaryGenerationBot,
)

class ReviewBot:
    task_loaders = {
        'search_bot': SearchBot.get_search_bot,
        'screener_bot': ScreenerBot.load_screener,
        'mds_summary_bot': MultiSummaryGenerationBot.load_mds_summary_bot,
    }

    def __init__(self, tasks=None, trial_reader=None, device='auto'):
        if tasks is None:
            tasks = ReviewBot.task_loaders.keys()
        else:
            assert all([task in ReviewBot.task_loaders for task in tasks])

        self.models = {task: ReviewBot.task_loaders[task](device=get_device(device)) for task in tasks}
        self.dynamically_switch_to_gpu = device == 'dynamic'

    def generate_search(topic: str) -> str:
        pass

    def execute_search(topic: str=None, query: str=None) -> List[Dict[str, str]]:
        assert topic is None != query is None

    def apply_screener(pubmed_results: List[Dict[str, str]], threshold=None):
        # TODO do we want any auto screener?
        pass

    def apply_trial_reader(pubmed_results: List[Dict[str, str]]):
        pass

class TrialReader:
    task_loaders = {
        "rct_bot": RCT_classifier.AbsRCTBot,
        "pico_span_bot": PICO_tagger.PICOBot,
        "punchline_bot": ev_inf_classifier.EvInfBot,
        "ico_ev_bot": ICOev_extractor.ICOBot,
        "bias_ab_bot": RoB_classifier_LR.AbsRoBBot,
        "sample_size_bot": sample_size_extractor.MLPSampleSizeClassifier,
        "study_design_bot": study_design_classifier.AbsStudyDesignBot,
        "numerical_extraction_bot": NumericalExtractionBot.get_numerical_extractor_bot,
    }


    def __init__(self, tasks=None, device='auto'):
        if tasks is None:
            tasks = TrialReader.task_loaders.keys()
        else:
            assert all([task in TrialReader.task_loaders for task in tasks])

        self.models = {task: TrialReader.task_loaders[task](device=get_device(device)) for task in tasks}
        self.dynamically_switch_to_gpu = device == 'dynamic'

    def read_trial(
            self,
            ab: dict,
            process_rcts_only=True,
            task_list=None,
            previous=None,
        ) -> Type[dict]:
        """
        The default behaviour is that non-RCTs do not have all extractions done (to save time).
        If you wish to use all the models anyway (which might not behave entirely as expected)
        then set `process_rcts_only=False`.
        """

        if task_list is None:
            task_list = ["rct_bot", "pico_span_bot", "punchline_bot",
                   "bias_ab_bot", "sample_size_bot"]
        # do not modify the source list which may get re-used many times.
        task_list = list(task_list)

        if previous is not None:
            return_dict = dict(copy.deepcopy(previous))
        else:
            return_dict = {}
            return_dict["rct_bot"] = {"is_rct": False}

        if "numerical_extraction_bot" in task_list:
            perform_numerical_extraction = True
            assert 'ico_ev_bot' in task_list, "Require ICO elements to perform numerical extraction"
            task_list.remove('numerical_extraction_bot')
        else:
            perform_numerical_extraction = False

        if process_rcts_only:
            task_list.remove('rct_bot')
            # First: is this an RCT? If not, the rest of the models do not make
            # a lot of sense so we will warn the user
            if 'rct_bot' not in return_dict:
                return_dict["rct_bot"] = self.models['rct_bot'].predict_for_ab(ab)

        if not return_dict["rct_bot"]["is_rct"]:
            if process_rcts_only:
                warnings.warn('''Predicted as non-RCT, so rest of models not run. Re-run
                         with `process_rcts_only=False` to get all predictions.''')
            else:
                warnings.filterwarnings('once', 'The input does not appear to describe an RCT;'
                        'interpret predictions accordingly.')

        if (not process_rcts_only) or return_dict["rct_bot"]["is_rct"]:
            for task in task_list:
                # skip the task if we have already done it
                if task not in return_dict:
                    if self.dynamically_switch_to_gpu and self.models[task].supports_gpu():
                        self.models[task].to('cuda')
                    return_dict[task] = self.models[task].predict_for_ab(ab)
                    if self.dynamically_switch_to_gpu and self.models[task].supports_gpu():
                        self.models[task].to('cpu')
            if perform_numerical_extraction and 'numerical_extraction_bot' not in return_dict:
                if self.dynamically_switch_to_gpu and self.models['numerical_extraction_bot'].supports_gpu():
                    self.models['numerical_extraction_bot'].to('cuda')
                return_dict['numerical_extraction_bot'] = self.models['numerical_extraction_bot'].predict_for_ab(ab, return_dict['ico_ev_bot'])
                if self.dynamically_switch_to_gpu and self.models['numerical_extraction_bot'].supports_gpu():
                    self.models['numerical_extraction_bot'].to('cpu')

        return return_dict


# For e.g.:
# import rrnlp
# trial_reader = rrnlp.TrialReader()

# ti_abs = {"ti": 'A Cluster-Randomized Trial of Hydroxychloroquine for Prevention of Covid-19',
#           "ab": """ Background: Current strategies for preventing severe acute
#            respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are
#            limited to nonpharmacologic interventions. Hydroxychloroquine has
#            been proposed as a postexposure therapy to prevent coronavirus
#            disease 2019 (Covid-19), but definitive evidence is lacking.

#           Methods: We conducted an open-label, cluster-randomized trial
#           involving asymptomatic contacts of patients with
#           polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia,
#           Spain. We randomly assigned clusters of contacts to the
#           hydroxychloroquine group (which received the drug at a dose of 800 mg
#           once, followed by 400 mg daily for 6 days) or to the usual-care
#           group (which received no specific therapy). The primary outcome was
#           PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary
#           outcome was SARS-CoV-2 infection, defined by symptoms compatible with
#           Covid-19 or a positive PCR test regardless of symptoms. Adverse
#           events were assessed for up to 28 days.\n\nResults: The analysis
#           included 2314 healthy contacts of 672 index case patients with
#           Covid-19 who were identified between March 17 and April 28, 2020. A
#           total of 1116 contacts were randomly assigned to receive
#           hydroxychloroquine and 1198 to receive usual care. Results were
#           similar in the hydroxychloroquine and usual-care groups with respect
#           to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and
#           6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52
#           to 1.42]). In addition, hydroxychloroquine was not associated with a
#           lower incidence of SARS-CoV-2 transmission than usual care (18.7% and
#           17.8%, respectively). The incidence of adverse events was higher in
#           the hydroxychloroquine group than in the usual-care group (56.1% vs.
#           5.9%), but no treatment-related serious adverse events were
#           reported.\n\nConclusions: Postexposure therapy with
#           hydroxychloroquine did not prevent SARS-CoV-2 infection or
#           symptomatic Covid-19 in healthy persons exposed to a PCR-positive
#           case patient. (Funded by the crowdfunding campaign YoMeCorono and
#           others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).
#           """
# }
# preds = trial_reader.read_trial(ti_abs)
