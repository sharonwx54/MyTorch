import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        sym_num, seq_len, batch_size = y_probs.shape
        full_symbol_set = [""]+self.symbol_set
        curr_idx = 0
        for s in range(seq_len):
            idx = np.argmax(y_probs[:, s, :], axis=0)[0]
            #print("the argmax index is "+str(idx))
            # update max prob
            path_prob *= y_probs[idx, s, :]
            if idx > 0:
                # for idx = 0, it is the blank
                # append if in the first step, or if symbol is not the same 
                if (len(decoded_path) == 0) or (curr_idx == 0) or (full_symbol_set[idx] != decoded_path[-1]):
                    decoded_path.append(full_symbol_set[idx]) 
            curr_idx = idx
        final_path = ""
        for p in decoded_path:
            final_path+=p

        decoded_path = final_path

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def prune(self, blank_score_dic, path_score_dic, blank_path, symbol_path):
        prune_blank_score_dic, prune_path_score_dic = {}, {}
        scores = [blank_score_dic[p][0] for p in blank_path]+[path_score_dic[p][0] for p in symbol_path]
        #print(scores)
        scores.sort(reverse=True) # sort in decreasing order


        threshold = scores[self.beam_width-1] if self.beam_width < len(scores) else scores[-1]

        prune_blank_path, prune_symbol_path = [], []
        for pb in blank_path:
            if blank_score_dic[pb] >= threshold:
                prune_blank_path.append(pb)
                prune_blank_score_dic[pb] = blank_score_dic[pb]
        for ps in symbol_path:
            if path_score_dic[ps] >= threshold:
                prune_symbol_path.append(ps)
                prune_path_score_dic[ps] = path_score_dic[ps]

        return prune_blank_score_dic, prune_path_score_dic, prune_blank_path, prune_symbol_path
    

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        blank_path, symbol_path = [], []
        best_path, final_path_score = None, None
        # initialization
        y_probs_0 = y_probs[:, 0, :]
        blank_score_dic, path_score_dic = {}, {}
        blank_score_dic[""] = y_probs_0[0]
        curr_sym = ""
        blank_path.append(curr_sym)
        for s in range(len(self.symbol_set)):
            curr_sym = self.symbol_set[s]
            # plus 1 as y_prob starts with blank
            path_score_dic[curr_sym] = y_probs_0[s+1]
            symbol_path.append(curr_sym)

        # iteration
        for t in range(1, T):
            prune_blank_dic, prune_path_dic, prune_blank_path, prune_symbol_path = self.prune(
                blank_score_dic, path_score_dic, blank_path, symbol_path)
            prune_symbol_path_extend = []
            prune_path_dic_extend = {}
            # updating blank/symbol on paths that end with blank
            for pb in prune_blank_path:
                pb_score = prune_blank_dic[pb]*y_probs[0, t] # 0 is prob for blank
                for s in range(len(self.symbol_set)):
                    s_concat = pb+self.symbol_set[s]
                    prune_symbol_path_extend.append(s_concat)
                    prune_path_dic_extend[s_concat] = prune_blank_dic[pb]*y_probs[s+1, t]
                # update blank dic value
                prune_blank_dic[pb] = pb_score
            # updating blank/symbol on paths that end with symbol
            for ps in prune_symbol_path:
                # for blank
                if ps in prune_blank_path:
                    prune_blank_dic[ps] += prune_path_dic[ps]*y_probs[0, t]
                else:
                    prune_blank_dic[ps] = prune_path_dic[ps]*y_probs[0, t]
                    prune_blank_path.append(ps)

                # for symbol
                for s in range(len(self.symbol_set)):
                    s_concat = ps 
                    if self.symbol_set[s] != ps[-1]:
                        s_concat += self.symbol_set[s]
                    if s_concat in prune_symbol_path_extend:
                        # if exist, add the prob
                        prune_path_dic_extend[s_concat] += prune_path_dic[ps]*y_probs[s+1, t]
                    else:
                        prune_path_dic_extend[s_concat] = prune_path_dic[ps]*y_probs[s+1, t]
                        prune_symbol_path_extend.append(s_concat)
            # update input for next loop
            blank_score_dic, path_score_dic = prune_blank_dic, prune_path_dic_extend
            blank_path, symbol_path = prune_blank_path, prune_symbol_path_extend

        # merging
        final_path_score = prune_path_dic_extend
        for pb in prune_blank_path:
            if pb in prune_symbol_path_extend:
                    final_path_score[pb] += prune_blank_dic[pb]
            else:
                    final_path_score[pb] = prune_blank_dic[pb]
                    prune_symbol_path_extend.append(pb)

        # select the best
        best_path = max(final_path_score, key=final_path_score.get)

        return best_path, final_path_score

