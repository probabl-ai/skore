<script setup lang="ts">
import { sleep } from "@/services/utils";

await sleep(2000);
</script>

<template>
  <main class="activity-feed">
    <h1>Activity feed</h1>
    <h2>Find all your activity, right below.</h2>
  </main>
</template>

<style scoped>
.activity-feed {
  color: red;
}
</style>
