export default {
    rules: {
    },
    ignores: [(commit) => commit.startsWith('chore: bump')],
};
