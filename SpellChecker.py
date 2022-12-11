import random
import math
from collections import defaultdict

# Import the Ngram language model module
import NgramLM


class Spell_Checker:
    def __init__(self, lm=None):
        """Initializing the spell checker object with a language model (LM) as an
        instance  variable. The LM should suppport the evaluate()
        and get_model() methods.

        Args:
            lm: a LM object. Defaults to None
        """
        self.lm = lm
        self.lm_vocab = None # class attribute: will be used to recreate lm vocabulary from lm.model_dict
        self.lm_vocab_size = None # class attribute: will be used to recreate total size (number of distinct words) of vocabulary in lm
        self.lm_vocab_count = None # class attribute: will be used to recreate total number (number of words with repetition) of vocabulary in lm
        self.lm_model_dict = None # class attribute: model_dict attribute from lm.get_model_dictionary()
        if lm:
            self.calculate_lm_attributes()
        self.error_tables = None # class attribute: will be used to store error tables
        self.count_xw = None # class attribute: will be used to store counts of char-char pairs (based on error_tables / lm_vocab)

    def build_model(self, text, n=3):
        """Returns a LM object built on the specified text.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model.

            Returns:
                A LM object
        """
        text = normalize_text(text)

        self.lm = Ngram_Language_Model(n=n, chars=False)
        self.lm.build_model(text)
        self.calculate_lm_attributes()

        return self.lm

    def calculate_lm_attributes(self):
        '''
        called when new lm is stored in class.
        calculates lm_vocab / lm_vocab_size / lm_vocab_count and stores them in class attributes
        '''
        self.lm_vocab = defaultdict(int)
        self.lm_model_dict = self.lm.get_model_dictionary()
        # recreates list of tokens (with duplicates) from lm_model_dict attribute
        grams_list = [ngram.split()[0] for ngram, count in self.lm_model_dict.items() for _ in range(count)]
        for word in grams_list:
            if self.lm_vocab.get(word):
                self.lm_vocab[word] += 1
            else:
                self.lm_vocab[word] = 1

        self.lm_vocab_size = len(self.lm_vocab)
        self.lm_vocab_count = len([w_count for w_count in self.lm_vocab.values()])

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                ls: a language model object
        """
        self.lm = lm
        # after storing new lm - recalculates lm attributes
        self.calculate_lm_attributes()

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary where str is in:
        <'deletion', 'insertion', 'transposition', 'substitution'> and the
        inner dict represents the confution matrix of the specific errors,
        where str is a string of two characters mattching the
        row and culumn "indixes" in the relevant confusion matrix and the int is the
        observed count of such an error.

        Examples of such string are:
        1. 'xy', for deletion of a 'y' after an 'x'
        2. insertion of a 'y' after an 'x'
        3. substitution of 'x' (incorrect) by a 'y'
        4. transposition is 'xy' indicates the characters that are transposed

        Args:
            errors_file (str): full path to the errors file. File format, TSV:
                                <error>    <correct>


        Returns:
            A dictionary of confusion "matrices" by error type.
        """
        # count_xw_from_corpus = True: counts for char-char are calculated from vocabulary
        # count_xw_from_corpus = False: counts for char-char are calculated  from error file
        count_xw_from_corpus = True

        self.error_tables = {
            "deletion": dict(),
            "insertion": dict(),
            "transposition": dict(),
            "substitution": dict()
        }

        with open(errors_file, 'r') as f:
            content = f.readlines()

        if count_xw_from_corpus and self.lm_vocab:
            correct_words_list = [('*' + w) for w, count in self.lm_vocab.items() for _ in range(count)]
        else:
            correct_words_list = ['*' + (l.split())[1].lower() for l in content]

        # creating count_xw dict ({xw: count, ...})
        self.count_xw = defaultdict(int)
        for word in correct_words_list:
            for i, l in enumerate(word):
                c1 = word[i]
                self.count_xw[c1] += 1
                if i < (len(word) - 1):
                    c2 = word[i+1]
                    self.count_xw[c1 + c2] += 1

        with open(errors_file, 'r') as f:
            error_lines = f.readlines()

        error_lines = [(e.split()) for e in error_lines]

        # initializing error tables
        for l1 in "*abcdefghijklmnopqrstuvwxyz'- ":
            for l2 in "*abcdefghijklmnopqrstuvwxyz'- ":
                self.error_tables["deletion"][l1 + l2] = 0
                self.error_tables["insertion"][l1 + l2] = 0
                self.error_tables["transposition"][l1 + l2] = 0
                self.error_tables["substitution"][l1 + l2] = 0

        for line in error_lines:
            error, correct = '*' + line[0].lower(), '*' + line[1].lower()

            # insertion error
            if len(error) > len(correct):
                correct += '~'
                for i in range(len(correct) - 1):
                    c1_err, c2_err = error[i], error[i+1]
                    c1_corr, c2_corr = correct[i], correct[i + 1]
                    if c2_err != c2_corr:
                        self.error_tables["insertion"][c1_corr + c2_err] += 1
                        break

            # deletion error
            elif len(error) < len(correct):
                error += '~'
                for i in range(len(correct) - 1):
                    c1_err, c2_err = error[i], error[i+1]
                    c1_corr, c2_corr = correct[i], correct[i + 1]
                    if c2_err != c2_corr:
                        self.error_tables["deletion"][c1_corr + c2_corr] += 1
                        break

            # transposition error
            elif sum([ord(c) for c in error]) == sum([ord(c) for c in correct]):
                for i in range(len(correct) - 1):
                    c1_err, c2_err = error[i], error[i+1]
                    c1_corr, c2_corr = correct[i], correct[i + 1]
                    if c2_err != c2_corr:
                        self.error_tables["transposition"][c2_corr + c2_err] += 1
                        break

            # substitution error
            else:
                for i in range(len(correct) - 1):
                    c1_err, c2_err = error[i], error[i+1]
                    c1_corr, c2_corr = correct[i], correct[i + 1]
                    if c2_err != c2_corr:
                        self.error_tables["substitution"][c2_err + c2_corr] += 1
                        break

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        # creating count_xw dict ({xw: count, ...}) in case doesn't exist
        if not self.count_xw:
            correct_words_list = [('*' + w) for w, count in self.lm_vocab.items() for _ in range(count)]
            self.count_xw = defaultdict(int)
            for word in correct_words_list:
                for i, l in enumerate(word):
                    c1 = word[i]
                    self.count_xw[c1] += 1
                    if i < (len(word) - 1):
                        c2 = word[i + 1]
                        self.count_xw[c1 + c2] += 1

        self.error_tables = error_tables

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the LM in use.
           Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text)

    def spell_check(self, text=None, alpha=None):
        """ Returns the most probable fix for the specified text.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        if not text or not alpha:
            raise Exception("text / alpha values were not defined.")
        # list of tokens that don't require spell_check (due to being punctuation / normalized_tokens)
        correction_unneeded = [".", ",", "?", "!", "(", ")", "SomeWeekday", "SomeMonth", "SomeYear", "SomeNumber",
                            "#Hashtag", "dd-mm-yyyy", "@TwitterUsername", "URL", "dd-mm", "$SomeMoney"]

        text = normalize_text(text)
        text_grams = text.split()
        corrections_prob = dict() # corrections_prob stores all optional corrections: {correction_candidate: log_likelihood}

        # for each grams -> replacing with possible candidates -> for each candidate -> calculate candidate probability for full sentence
        for i, gram in enumerate(text_grams):
            if gram in correction_unneeded:
                continue
            candidates = self.find_candidate_words(gram, alpha)
            for c, prob_xw in candidates.items():
                text_candidate = ' '.join(text_grams[:i]) + ' ' + c + ' ' + ' '.join(text_grams[i+1:])
                # text_candidate2 = text.replace(gram, c)
                text_prior = self.evaluate(text_candidate)
                ll_text = math.log(prob_xw) + text_prior
                corrections_prob[text_candidate] = ll_text

        # selecting correction with maximal probability
        corrected_text = max(corrections_prob, key=corrections_prob.get)

        return corrected_text

    def find_candidate_words(self, word, alpha):
        '''
        returns a dict ({candidate: prob_xw, ...}) of all candidate words that are 1 edit distance from word
        assumption taken:
        - errors that doesn't exist in the error tables / count_xw tables - is skipped (and not smoothed!)
        '''
        all_candidates = list()
        v = {'xy': 'original', 'word': word, 'prob_xw': alpha}
        all_candidates.append(v)

        # going through each table and creating adding possible candidates to 'all_candidates'
        for err_type, err_table in self.error_tables.items():
            if err_type == "deletion":
                # deletion of a 'y' after an 'x'
                for i in range(len(word) + 1):
                    w = list('*' + word)
                    for del_c in "abcdefghijklmnopqrstuvwxyz'- ":
                        new_w = w.copy()
                        x = new_w[i]
                        new_w.insert(i+1, del_c)
                        xy = x + del_c
                        if self.count_xw.get(xy) and err_table.get(xy):
                            del_xy = err_table[xy]
                            count_xy = self.count_xw[xy]
                            xw = ''.join(new_w[1:])
                            v = {'xy': xy, 'word': xw, 'prob_xw': (1 - alpha) * (del_xy / count_xy)}
                            if v['prob_xw']:
                                all_candidates.append(v)

            elif err_type == "insertion":
                #  insertion of a 'y' after an 'x'
                for i in range(1, len(word) + 1): # hy instead of h
                    w = list('*' + word)
                    x = w[i-1]
                    y = w.pop(i)
                    xy = x + y
                    if self.count_xw.get(x) and err_table.get(xy):
                        ins_xy = err_table[xy]
                        count_x = self.count_xw[x]
                        xw = ''.join(w[1:])
                        v = {'xy': xy, 'word': xw, 'prob_xw': (1 - alpha) * (ins_xy / count_x)}
                        if v['prob_xw']:
                            all_candidates.append(v)

            elif err_type == "transposition":
                # transposition is 'xy' indicates the characters that are transposed
                for i in range(len(word) - 1):
                    w, c1, c2 = list(word), word[i], word[i+1]
                    w[i], w[i+1] = c2, c1
                    xy = c2 + c1
                    if self.count_xw.get(xy) and err_table.get(xy):
                        trans_xy = err_table[xy]
                        count_xy = self.count_xw[xy]
                        xw = ''.join(w)
                        v = {'xy': xy, 'word': xw, 'prob_xw': (1 - alpha) * (trans_xy / count_xy)}
                        if v['prob_xw'] and self.lm_vocab.get('word'):
                            all_candidates.append(v)

            elif err_type == "substitution":
                # substitution of 'x' (incorrect) by a 'y'
                for i in range(len(word)):
                    err_c = word[i]
                    for y in "abcdefghijklmnopqrstuvwxyz'- ":
                        w = list(word)
                        w[i] = y
                        xy = err_c + y
                        if self.count_xw.get(y) and err_table.get(xy):
                            sub_xy = err_table[xy]
                            count_y = self.count_xw[y]
                            xw = ''.join(w)
                            v = {'xy': xy, 'word': xw, 'prob_xw': (1 - alpha) * (sub_xy / count_y)}
                            if v['prob_xw'] and self.lm_vocab.get('word'):
                                all_candidates.append(v)

        # summing probabilities for multiple entries (i.e., if 'ab' appears in transposition and insertion)
        candidates = dict()
        for c in all_candidates:
            if candidates.get(c['word']):
                candidates[c['word']] += c['prob_xw']
            else:
                candidates[c['word']] = c['prob_xw']

        # keeping only candidates that have prob_xw > 0 and exist in lm_vocab
        candidates = {k: v for k, v in candidates.items() if v and self.lm_vocab.get(k)}

        return candidates


def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Assaf Shamir', 'id': '021645619', 'email': 'shamiras@post.bgu.ac.il'}