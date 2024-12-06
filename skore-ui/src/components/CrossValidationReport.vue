<script setup lang="ts">
import TabPanel from "@/components/TabPanel.vue";
import TabPanelContent from "@/components/TabPanelContent.vue";
</script>

<template>
  <div class="cross-validation-report">
    <TabPanel>
      <TabPanelContent name="results"></TabPanelContent>
      <TabPanelContent name="plots"></TabPanelContent>
      <TabPanelContent name="details"></TabPanelContent>
    </TabPanel>
  </div>
</template>
