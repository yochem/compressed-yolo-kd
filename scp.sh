#!/usr/bin/env bash

changed=$(git status --porcelain | cut -c 4-)

scp_path="Yochem@cpu8gpu2:~/yolokd/compressed-yolo-kd"

for file in $changed; do
	gcloud compute scp "$file" "$scp_path/$file" --recurse --zone "europe-west4-c"
	git add "$file"
done

git commit -m "scp to server" >/dev/null
