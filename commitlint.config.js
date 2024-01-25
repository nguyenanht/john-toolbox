module.exports = {
    rules: {
    },
    ignores: [(commit) => commit.startsWith('chore: bump')],
};
