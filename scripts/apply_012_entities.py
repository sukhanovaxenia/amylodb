"""Apply the PUM, HNRNPH1 and RepA curator decisions to the audited v3h snapshot."""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

SHA = "31bbe377f7523d3b4567280374909120f738c0555ffa5a257e561c376ef42539"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def apply(source, output):
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        target = Path(tmp)/"stage.db"
        shutil.copyfile(source,target)
        require(hashlib.sha256(target.read_bytes()).hexdigest()==SHA,"Wrong input snapshot")
        with closing(sqlite3.connect(target)) as db, db:
            def note(pid, action, evidence):
                item=dict(source_row_id=pid, decision=action, evidence=evidence, stage="012")
                changed=db.execute("UPDATE proteins SET curator_decision=? WHERE id=? AND coalesce(curator_decision,'')=''",(json.dumps(item),pid))
                require(changed.rowcount==1,f"Existing decision or absent row {pid}")
            def link(pid, pmid):
                db.execute("INSERT INTO protein_papers(protein_id,pmid) VALUES (?,?)",(pid,pmid))
            db.execute("PRAGMA foreign_keys=ON")
            pid=db.execute("""INSERT INTO proteins(protein_name,gene_name,organism,uniprot_id,uniprot_name,resolved_name,resolved_organism,status,protein_role,forms_aggregates,condensate_evidence,modifier_roles)
                VALUES ('Pumilio homolog 1','PUM1','Homo sapiens','Q14671','Pumilio homolog 1','Pumilio homolog 1','Homo sapiens','kept','condensate',0,1,'[]')""").lastrowid
            for dest in [pid,1194]:
                link(dest,'25340845')
                note(dest,'condensate_component',{'25340845':'Granule recruitment; resolved from row 1416'})
            note(1416,'split_generic_claim',{'25340845':[pid,1194]})
            db.execute("""INSERT INTO papers(pmid,title,doi,year) VALUES ('27623008','The C9ORF72 GGGGCC expansion forms RNA G-quadruplex inclusions and sequesters hnRNP H to disrupt splicing in ALS brains','10.7554/eLife.17820',2016)""")
            link(1062,'27623008')
            note(1062,'retain_condensate_domain_evidence',{'34873036':'LC1 reversible polymerization and phase separation','27623008':'Insoluble hnRNP H in G-quadruplex-associated assemblies; no assignment of full-length amyloid'})
            note(1662,'retain_condensate_component',{'32473225':'Stress granule localization'})
            note(1661,'retain_dropped_mixed_entity',{'32473225':'Already represented by row 1662','35340804':'hnRNP K; separate attribution required'})
            scopes={'17959784':'WH1 domain','25543255':'WH1 domain','27147472':'full-length RepA assembly','24794561':'WH1 and variants; E. coli is experimental host'}
            for rid in (387,592,668):
                pmids=[str(p[0]) for p in db.execute('select pmid from protein_papers where protein_id=?',(rid,))]
                evidence={p:scopes[p] for p in pmids}
                db.execute("""UPDATE proteins SET uniprot_id='Q52546',gene_name='repA',uniprot_name='Replication protein RepA',resolved_name='Replication protein RepA',resolved_organism='Pseudomonas syringae (plasmid pPS10)',match_type='curator_review',lookup_stage='curator_review',resolved_from='pPS10 RepA' WHERE id=?""",(rid,))
                note(rid,'correct_identity_pPS10',evidence)
            require(db.execute('pragma integrity_check').fetchone()==('ok',),'Integrity failure')
        os.link(target,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    apply(args.db,args.out)
