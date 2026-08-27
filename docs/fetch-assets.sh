#!/usr/bin/env bash
# Downloads the AI-generated engraving artwork into docs/assets/.
# Run once from anywhere, then commit the result:
#
#   bash docs/fetch-assets.sh
#   git add docs/assets && git commit -m "Add generated site artwork" && git push
#
# NOTE: the URLs below are signed and expire around 2026-09-02. If they have
# expired, regenerate the artwork with the prompts in docs/assets/PROMPTS.md.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets

curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2F0e4b8d0593ed01ce4637409f2a23cc0bd2d33b4d82ef2c76482c7957fdc1bdea?filename=hero.jpg&sig=IfQcjbVNyB_ndu1qnyjpPWiT6YMbNXE68FlC7F3n5as=&t=o" -o assets/hero.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2F97758c113bb723d79ceea8b7e43eec55b761cfe3a9a8d303b0539ecd8af327c6?filename=feat_batch.jpg&sig=A3WoIKoZFw60F-6RQU4LsZGm1WTTUfCn_AzwqQSxe74=&t=o" -o assets/feat_batch.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2Ffa1626edd3f255493e2746c0021fe30eb27765b949eaadb700c024c8c3d7e610?filename=feat_resume.jpg&sig=pOsQSE5Q76nJqHFxe-aHF2r98RTFwbzOzJ-BwA8up3k=&t=o" -o assets/feat_resume.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2F5de3c3ab6cd9a5a058d8f36125d05d482470fede705c8222b2c69bbd47d85f6c?filename=feat_score.jpg&sig=cgPHK2ANLGiJLjRmvAyI1JDSKQpiP_6pPlh-sYJnkco=&t=o" -o assets/feat_score.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2F31dd8519bf96cd883c69954f8fee596bbac427256c37e7a9c52438d04dc54078?filename=feat_automate.jpg&sig=9q55D_SdCF1X9NlE9Hev5QfeME_jZbx9FE61TyQGco8=&t=o" -o assets/feat_automate.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2Fab2883cb89507cb1f7ecedb586fba0b7ee9f07e1d69ea6dce878260d537d0192?filename=feat_train.jpg&sig=wt2Rm8yKJ-Kg9zf0rMDl_OeNAUT6S7Wab-QeTBMP8hY=&t=o" -o assets/feat_train.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2F72dbde66d0c8fee687f38d5eb118433068bb74dfa29847fe93665ad110c340be?filename=feat_backends.jpg&sig=lzWDcuskgwSAA0fPLaWY2XtI0gFpv6PYByrf6JZ-_ac=&t=o" -o assets/feat_backends.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2F01083871ce70f2af2e1fa180e6cde335a829094d91bf5210c0e169e8fe8436e6?filename=footer_art.jpg&sig=ORbtNFmcBP9eRBBp6gf-n3_eHkViYthamdyjLltXf0c=&t=o" -o assets/footer_art.jpg
curl -fsSL "https://www.kimi.com/apiv2-files/sign-obj/kimi-fs%2Ffiles%2Fblob%2F58c6b45a235bea2b363e8a9676634d402282e1949b930f827848b7a5233d39ad?filename=mark.jpg&sig=7RYXa4wi81qMkfvBK_3XW6r8bBNYMudAgkfCt7z1Dlk=&t=o" -o assets/mark.jpg

echo "Done — artwork saved to docs/assets/"
ls -la assets/
