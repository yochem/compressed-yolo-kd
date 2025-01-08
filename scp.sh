set -o nounset

for file in "$@"; do
	gcloud compute scp $file "Yochem@cpu8gpu2:~/yolokd/repo/$file" --project "afstuderen-yochem" --recurse --zone "europe-west4-c"
done

git add $@
git commit -m "scp to server" >/dev/null
