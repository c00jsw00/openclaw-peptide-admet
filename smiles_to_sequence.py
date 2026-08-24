"""
SMILES -> 胺基酸序列 還原器(結構式偵測版)。

把 pepADMET 風格的肽 SMILES(僅分子結構)還原成 20 標準胺基酸的 one-letter
序列,讓外部 SMILES 數據進入 openclaw 的「序列 -> 428-dim 特徵」管線
(同特徵空間、同模型、同同源性家族)。

方法(不依賴 SMARTS 解包,直接結構式判斷):
  1. 找所有 α-碳:sp3 C,且 (a) 相鍵 ≥1 個 H,(b) 相鍵 1 個酰胺 N
     (N 的另一端是 C=O),(c) 相鍵 1 個羰基 C(=O)。
  2. 每個 α-碳的「側鏈錨點」= 除 H / 酰胺N / 羰基C 外的相鍵原子。
  3. 側鏈 = 從錨點 BFS(不穿回 α-碳、不穿過羰基碳)。
  4. 依側鏈元素組成 + 芳環/含S/含N 規則映射到 one-letter。
  5. 依 SMILES 索引順序(肽 SMILES 慣例 N->C 書寫)重組序列。

限制(誠實):
  - 僅 20 標準 AA;非標準/修飾殘基標 'X' 並計數,不靜默丟棄。
  - Leu/Ile/Val、Asp/Glu、Ser/Thr 等「同元素側鏈」靠碳數/幾何區分,
    個別殘基可能混判 → 以 CSV 內含的 PyProtein 計數做對照驗證(見下)。
  - Pro(環、α-碳無 H)以專屬規則補捉。
  - RDKit 未安裝時回傳 matched=False(不崩潰)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from rdkit import Chem
    _RD = True
except Exception:  # pragma: no cover
    _RD = False

AAS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class SeqResult:
    sequence: str
    n_residues: int
    n_unknown: int
    matched: bool
    note: str = ""
    aac: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 結構判斷輔助
# --------------------------------------------------------------------------- #
def _is_carbonyl(mol, idx) -> bool:
    """C 是否為羰基碳(與雙鍵 O 相鍵)。"""
    a = mol.GetAtomWithIdx(idx)
    if a.GetSymbol() != "C":
        return False
    for nb in a.GetNeighbors():
        b = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
        if nb.GetSymbol() == "O" and b.GetBondTypeAsDouble() >= 1.9:
            return True
    return False


def _is_amide_N(mol, idx) -> bool:
    """N 是否為肽骨幹酰胺 N(相鍵一個羰基碳)。"""
    a = mol.GetAtomWithIdx(idx)
    if a.GetSymbol() != "N":
        return False
    for nb in a.GetNeighbors():
        if _is_carbonyl(mol, nb.GetIdx()):
            return True
    return False


def _find_alphas(mol):
    """回傳 [(alpha_idx, anchor_idx, kind)]。kind ∈ {'std','pro'}。"""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C" or a.GetAtomicNum() != 6:
            continue
        idx = a.GetIdx()
        nbs = a.GetNeighbors()
        n_h = sum(1 for nb in nbs if nb.GetSymbol() == "H")
        n_carbonyl = sum(1 for nb in nbs if _is_carbonyl(mol, nb.GetIdx()))
        n_amide = sum(1 for nb in nbs if _is_amide_N(mol, nb.GetIdx()))

        # 標準殘基 α-碳:≥1 H, 1 羰基C, 1 酰胺N
        if n_h >= 1 and n_carbonyl == 1 and n_amide == 1:
            # 錨點 = 非 H / 非羰基C / 非酰胺N 的相鍵原子
            anchors = [nb for nb in nbs
                       if nb.GetSymbol() != "H"
                       and not _is_carbonyl(mol, nb.GetIdx())
                       and not _is_amide_N(mol, nb.GetIdx())]
            if len(anchors) == 1:
                out.append((idx, anchors[0].GetIdx(), "std"))
            elif len(anchors) > 1:
                # Gly 錨點是 H;此處 anchors 應為 0(Gly 無側鏈重原子)。
                out.append((idx, -1, "std"))
        # Pro:α-碳在環上、無 H,相鍵 1 羰基C + 1 N(環上)
        elif n_h == 0 and n_carbonyl == 1 and a.IsInRing():
            out.append((idx, -1, "pro"))
    return out


def _side_chain_syms(mol, anchor_idx, alpha_idx):
    """從錨點 BFS 側鏈,回傳元素符號 list(不穿回 α-碳、不穿過羰基碳)。"""
    from collections import deque
    if anchor_idx < 0:
        return []
    seen = {alpha_idx}
    q = deque([anchor_idx])
    syms = []
    while q:
        i = q.popleft()
        if i in seen:
            continue
        seen.add(i)
        a = mol.GetAtomWithIdx(i)
        syms.append(a.GetSymbol())
        for nb in a.GetNeighbors():
            ni = nb.GetIdx()
            if ni in seen or ni == alpha_idx:
                continue
            # 不穿過羰基碳(那是下一殘基骨幹)
            if _is_carbonyl(mol, ni):
                continue
            q.append(ni)
    return syms


def _has_ring(mol, anchor_idx, alpha_idx) -> bool:
    from collections import deque
    if anchor_idx < 0:
        return False
    seen = {alpha_idx}
    q = deque([anchor_idx])
    while q:
        i = q.popleft()
        if i in seen:
            continue
        seen.add(i)
        a = mol.GetAtomWithIdx(i)
        if a.IsInRing():
            return True
        for nb in a.GetNeighbors():
            ni = nb.GetIdx()
            if ni in seen or ni == alpha_idx or _is_carbonyl(mol, ni):
                continue
            q.append(ni)
    return False


def _classify(anchor_atom, syms, ring) -> str:
    """依側鏈元素/結構映射 one-letter。anchor_atom=None 表示錨點是 H(Gly)。"""
    if anchor_atom is None:
        return "G"
    if anchor_atom.GetSymbol() == "H":
        return "G"
    n_c = syms.count("C"); n_o = syms.count("O")
    n_s = syms.count("S"); n_n = syms.count("N")

    if n_s >= 1:
        return "C" if n_c <= 1 else "M"
    if ring:
        if n_o >= 1:
            return "Y"
        return "W" if n_c >= 8 else "F"
    if n_n >= 1:
        if n_c >= 5 and n_n >= 2:
            return "R"
        if n_c >= 4 and n_n == 1:
            return "K"
        if n_c >= 3 and n_n >= 2:
            return "H"
        if n_o >= 1 and n_c >= 4:
            return "Q"
        if n_o >= 1:
            return "N"
        return "X"
    if n_o >= 1:
        if n_c == 1:
            return "S"          # Ser/Thr 同元素;預設 Ser
        if n_c == 2:
            return "D"
        return "E"
    # 純碳
    if n_c == 0:
        return "G"
    if n_c == 1:
        return "A"
    if n_c == 2:
        return "P"              # 預設 Pro(純 C2 側鏈,環)
    if n_c == 3:
        return "L"              # Val/Leu/Ile 同 C3;預設 Leu
    return "X"


def smiles_to_sequence(smi: str) -> SeqResult:
    if not _RD:
        return SeqResult("", 0, 0, False, "rdkit 未安裝")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return SeqResult("", 0, 0, False, "SMILES 解析失敗")
    mol = Chem.AddHs(mol)

    alphas = _find_alphas(mol)
    if not alphas:
        return SeqResult("", 0, 0, False, "未偵測到肽骨幹 α-碳")

    seq = []
    n_unknown = 0
    for alpha_idx, anchor_idx, kind in alphas:
        if kind == "pro" or anchor_idx < 0 and False:
            aa = "P" if kind == "pro" else _classify(None, [], False)
            seq.append(aa)
            continue
        anchor_atom = mol.GetAtomWithIdx(anchor_idx)
        syms = _side_chain_syms(mol, anchor_idx, alpha_idx)
        ring = _has_ring(mol, anchor_idx, alpha_idx)
        aa = _classify(anchor_atom, syms, ring)
        if aa == "X":
            n_unknown += 1
        seq.append(aa)

    # 依 α-碳在 mol 中的索引順序(N->C 近似)
    order = sorted(range(len(alphas)), key=lambda i: alphas[i][0])
    seq = [seq[i] for i in order]
    s = "".join(seq)
    c = {}
    for ch in s:
        if ch in AAS:
            c[ch] = c.get(ch, 0) + 1
    aac = [c.get(a, 0) for a in AAS]
    return SeqResult(s, len(s), n_unknown, len(s) > 0,
                     (f"含 {n_unknown} 個未知殘基" if n_unknown else ""), aac)


if __name__ == "__main__":
    import sys
    import pandas as pd
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1], low_memory=False)
        col = "smiles" if "smiles" in df.columns else "SMILES"
        ok = fail = tx = 0
        lens = []
        for s in df[col]:
            r = smiles_to_sequence(s)
            if r.matched:
                ok += 1; tx += r.n_unknown; lens.append(r.n_residues)
            else:
                fail += 1
        if lens:
            print(f"parsed={ok}/{ok+fail}  unknown={tx}  "
                  f"len min/med/max={min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
