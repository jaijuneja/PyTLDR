# encoding=utf-8
import preprocess
from scipy.sparse.linalg import svds
import numpy as np


def compute_svd(matrix, num_concepts=5):
    u, s, v = svds(matrix, k=num_concepts)
    return u, s, v


def lsa_steinberger(text):
    sentences, unprocessed_sentences = preprocess.tokenize_sentences(text)
    sentence_mat = preprocess.compute_frequency_matrix(sentences)

    # Filter out negatives in the sparse matrix (need to do this on Vt for LSA method):
    sentence_mat = sentence_mat.multiply(sentence_mat > 0)

    s, u, v = compute_svd(sentence_mat)

    # Only consider topics/concepts whose singular values are half of the largest singular value
    sigma_threshold = max(u) / 2
    u[u < sigma_threshold] = 0  # Set all other singular values to zero

    # Build a "length vector" containing the length (i.e. saliency) of each sentence
    length_vec = np.dot(np.square(u), np.square(v))

    top_sentences = length_vec.argsort()[-5:][::-1]
    top_sentences.sort()

    return [unprocessed_sentences[i] for i in top_sentences]


def lsa_ozsoy(text):
    pass


if __name__ == "__main__":
    txt = """
    Sixty-four Islamic State fighters have been killed and dozens wounded in Egyptian-Libyan military airstrikes on Libya, announced the spokesperson of the Libyan military, reported Al-Ahram.

    The airstrikes, Egypt’s first foreign strikes since the Gulf War in 1991, come after ISIS released a video depicting the decapitation of 21 Egyptian Coptic Christians that had been abducted in Libya.

    According to Egypt’s military, the airstrikes were carried out in coordination with Libyan officials and targeted multiple ISIS locations, including weapons storage facilities, training points and buildings housing militants.

    Government officials have said that the airstrikes are the first of several to come which will aim to target terrorist elements in Libyan territory, said state media Al-Ahram.

    Earlier on Monday, Egypt’s President Abel Fattah Al-Sisi declared that Egypt had every right to respond after the brutal killings of the 21 Egyptians.

    Shortly after the President’s statement, the Military released video showing fighter jets taking off from Egypt at dawn.

    Despite reports in August 2014 that Egypt had carried out airstrikes on targets in Libya, these reports were ultimately denied by Egypt’s government and the US government issued a retraction.

    LOCAL, GLOBAL OUTRAGE

    On Monday, Egypt’s government announced that the families of the 21 Egyptians killed in Libya would receive EGP 100,000, a monthly ‘martyrs’ stipend, and health and educational support.

    The decision came after Egypt’s President Sisi and Prime Minister Ibrahim Mehleb visited the Abbassiya Cathedral to pay their respects.

    Meanwhile, across the globe, world leaders have condemned the killing of the 21 Egyptians. The White House described the beheadings as “cowardly” while France’s President reaffirmed France’s commitment to fighting ISIS.

    In a statement, the United Nations Security Council also condemned the deaths and stressed the importance of fighting terrorism.

    Palestinian, Sudanese, Emirati, Saudi, Jordanian and other Arab officials have also condemned the killings.

    EGYPT CALLS ON US-LED ISIS COALITION TO BATTLE ISIS IN LIBYA

    Egypt’s Ministry of Foreign Affairs stressed Egypt’s right to respond to the threat in Libya and called on the US-led anti-ISIS coalition to fight ISIS in Libya.

    In a statement, the Ministry of Foreign Affairs said “leaving matters inside Libya the way they are, without a strict intervention to curb these terrorist organisations, is a clear threat to international security and peace.”

    The statement comes as Egypt’s Minister of Foreign Affairs left Cairo for Washington DC on a five day visit.

    Meanwhile, both France’s President Francois Hollande and Egypt’s President Abdel Fattah Al-Sisi have called on the UN Security Council to take ‘new measures’ to battle ISIS in Libya.
    """.decode('utf-8')

    summary = lsa_steinberger(txt)

    for sentence in summary:
        print sentence